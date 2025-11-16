%% ========================================================================
%  Fourier-Based Inundation Estimation Module (FBIEM)
%  ------------------------------------------------------------------------
%  This script reconstructs the multi-temporal MNDWI curve for each pixel,
%  applies Savitzky–Golay filtering, and fits a two-term Fourier model to
%  extract long-term water presence parameters (a0–b2).
%
%  NOTE:
%  -----
%  Replace <your_mndwi_folder> and <your_output_folder> with your paths.
%
%  Output:
%     Five GeoTIFF files: A0.tif, A1.tif, B1.tif, A2.tif, B2.tif
%
%  Corresponds to Section II-C in the WetFramework paper.
%  ========================================================================
clear; clc;
warning('off','all');

%% ------------------------------------------------------------------------
% 1. Specify input/output folders
% -------------------------------------------------------------------------
inputFolder  = '<your_mndwi_folder>';       % Folder containing MNDWI *.tif files
outputFolder = '<your_output_folder>';      % Folder to save output coefficient maps

if ~exist(inputFolder, 'dir')
    error('Input folder not found. Please set inputFolder correctly.');
end
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

files = dir(fullfile(inputFolder, '*.tif'));
fileCount = numel(files);
if fileCount == 0
    error('No .tif files found in the input folder.');
end

%% ------------------------------------------------------------------------
% 2. Read spatial metadata from the first image
% -------------------------------------------------------------------------
[sampleImg, R] = geotiffread(fullfile(inputFolder, files(1).name));
info = geotiffinfo(fullfile(inputFolder, files(1).name));

[m, n] = size(sampleImg);
N = m * n;

% Pre-allocate full time series array
MNDWI = zeros(N, fileCount);
DOY   = zeros(1, fileCount);

fprintf('Loading multi-temporal MNDWI images...\n');

for i = 1:fileCount
    img = double(importdata(fullfile(inputFolder, files(i).name)));
    MNDWI(:, i) = img(:);

    % Parse DOY (Assuming filename format contains DOY at position 5:7)
    DOY(i) = str2double(files(i).name(5:7));
end

%% ------------------------------------------------------------------------
% 3. Pre-allocate Fourier coefficient maps
% -------------------------------------------------------------------------
A0 = zeros(N,1);
A1 = zeros(N,1);
B1 = zeros(N,1);
A2 = zeros(N,1);
B2 = zeros(N,1);

processed = 0;

% Target time points (24 samples per year)
xq = 15:15:360;
w = 2 * pi / 365;   % Annual frequency

%% ------------------------------------------------------------------------
% 4. S-G filtering + Fourier fitting for each pixel
% -------------------------------------------------------------------------
fprintf('\nPerforming S-G smoothing and Fourier fitting...\n');

for p = 1:N
    y = MNDWI(p, :);
    validIdx = ~isnan(y);

    if sum(validIdx) < 7
        continue;   % Not enough valid points
    end

    x_obs = DOY(validIdx);
    y_obs = y(validIdx);

    % ---- Step 1: Savitzky–Golay smoothing ----
    y_sg = sgolayfilt(y_obs, 4, 7);

    % ---- Step 2: Interpolate smoothed curve ----
    y_interp = interp1(x_obs, y_sg, xq, 'spline');

    % ---- Step 3: Two-term Fourier fitting ----
    params0 = [0 0 0 0 0];   % Initial guess
    params_fit = lsqcurvefit(@(params,x) Fourier2(params, x, w), ...
                             params0, xq, y_interp);

    % Store coefficients
    A0(p) = params_fit(1);
    A1(p) = params_fit(2);
    B1(p) = params_fit(3);
    A2(p) = params_fit(4);
    B2(p) = params_fit(5);

    processed = processed + 1;
    if mod(processed, 20000) == 0
        fprintf('Processed %d / %d pixels...\n', processed, N);
    end
end

%% ------------------------------------------------------------------------
% 5. Export GeoTIFF coefficient maps
% -------------------------------------------------------------------------
fprintf('\nSaving Fourier coefficient GeoTIFFs...\n');

geotiffwrite(fullfile(outputFolder,'A0.tif'), reshape(A0, m, n), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

geotiffwrite(fullfile(outputFolder,'A1.tif'), reshape(A1, m, n), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

geotiffwrite(fullfile(outputFolder,'A2.tif'), reshape(A2, m, n), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

geotiffwrite(fullfile(outputFolder,'B1.tif'), reshape(B1, m, n), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

geotiffwrite(fullfile(outputFolder,'B2.tif'), reshape(B2, m, n), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

fprintf('All coefficient maps saved successfully.\n');


%% ========================================================================
%  Two-term Fourier model (used in FBIEM)
% ========================================================================
function y = Fourier2(params, x, w)
    a0 = params(1);
    a1 = params(2);
    b1 = params(3);
    a2 = params(4);
    b2 = params(5);

    y = a0 ...
        + a1*cos(w*x)   + b1*sin(w*x) ...
        + a2*cos(2*w*x) + b2*sin(2*w*x);
end
