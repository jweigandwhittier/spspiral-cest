function convert_pge2(seq_filename, sys, pislquant, flag_plot, psd_rf_wait, psd_grd_wait, coil)
% Converts a Pulseq file (.seq) to GE format (.pge)
%
%
% INPUTS:
%   seq_filename  - Path to the .seq file to be converted
%   sys           - Pulseq system limits object
%   cvs           - Structure containing GE CVs (e.g., cvs.xloc, cvs.yloc, cvs.zloc)
%   flag_plot     - Boolean (true/false) to plot the sequence
%   psd_rf_wait   - (optional) RF wait timing compensation [s]. Defaults to
%                   the UCSF Premier value (148e-6) if omitted/empty.
%   psd_grd_wait  - (optional) Gradient wait timing compensation [s]. Defaults
%                   to the UCSF Premier value (152e-6) if omitted/empty.
%   coil          - (optional) Receive coil string for pge2.opts(). Defaults
%                   to 'hrmw' (UCSF Premier) if omitted/empty.
    
    %% Suppress warning for now
    warning('off', 'mr:restoreShape'); % Lots of apparently spiral-related warnings pop up

    %% Add necessary pge2/PulSeg
    % Extract file path base info
    [filepath, base_name, ~] = fileparts(seq_filename);
    % Get the absolute directory where THIS function lives
    this_dir = fileparts(mfilename('fullpath'));
    % Force absolute paths so the MATLAB engine doesn't get lost
    pge_path = fullfile(this_dir, '../pge2/matlab');
    addpath(genpath(pge_path));
    % Use genpath to recursively add all subfolders (including third_party!)
    pulseg_path = fullfile(this_dir, '../PulSeg/matlab');
    addpath(genpath(pulseg_path));
    % Look for the Pulseq toolbox
    pulseq_path = fullfile(this_dir, '../pulseq-1.5.1/matlab');
    if ~contains(path, 'pulseq-1.5.1/matlab')
        addpath(genpath(pulseq_path)); 
    end

    %% Convert to pge2 format
    % Execute the conversion
    fprintf('Converting %s to pge2 format...\n', seq_filename);
    
    psg = pulseg.fromSeq(seq_filename); % Convert to PulSeg

    % Define hardware parameters from main script
    % psd_rf_wait/psd_grd_wait/coil are site-specific (Berkeley vs UCSF)
    % deployment parameters, passed in as their own arguments (not part of
    % the Pulseq `sys` limits object). Fall back to the UCSF Premier values
    % that were previously hardcoded here.
    if nargin < 5 || isempty(psd_rf_wait)
        psd_rf_wait = 148e-6; % UCSF Premier fallback
    end
    if nargin < 6 || isempty(psd_grd_wait)
        psd_grd_wait = 152e-6; % UCSF Premier fallback
    end
    if nargin < 7 || isempty(coil)
        coil = 'hrmw'; % UCSF Premier fallback
    end
    b1_max = 0.25; % [Gauss], should be high enough to handle SPSP pulses?
    g_max = sys.maxGrad/sys.gamma*100;
    slew_max = sys.maxSlew/sys.gamma/10;

    sys_ge = pge2.opts(psd_rf_wait, psd_grd_wait, b1_max, g_max, slew_max, coil);

    % Check sequence
    PNSwt = [1 1 1];   % Directional PNS weights, see pge2.pns()
    params = pge2.check(psg, sys_ge, 'PNSwt', PNSwt);

    % Validate versus Pulseq file
    sys_pp = mr.opts( ...
    'MaxGrad', sys.maxGrad, ...
    'MaxSlew', sys.maxSlew, ...
    'GradRasterTime', sys.gradRasterTime, ...
    'RfRasterTime', sys.rfRasterTime, ...
    'AdcRasterTime', sys.adcRasterTime, ...
    'RfDeadTime', sys.rfDeadTime, ...
    'RfRingdownTime', sys.rfRingdownTime, ...
    'AdcDeadTime', sys.adcDeadTime, ...
    'BlockDurationRaster', sys.blockDurationRaster);
    seq = mr.Sequence(sys_pp);
    seq.read(seq_filename); % Turns out you actually have to load it like this??
    pge2.validate(psg, sys_ge, seq, [], 'row', [], 'plot', false);

    % Finally, convert
    out_filename = fullfile(filepath, [base_name '.pge']);
    pge2.serialize(psg, out_filename, 'pislquant', pislquant, 'params', params, 'checkHash', false);
    fprintf('Saved GE file to: %s\n', out_filename);

    % Also save the .mat file (psq/params/pislquant) needed by the
    % PulSeg/pge2 FOV-shift workflow (pulseq_shift_fov.sh -> translateFOVrf.m)
    % Variable must be named 'psq' on disk, per that workflow's convention
    psq = psg;
    mat_filename = fullfile(filepath, [base_name '.mat']);
    save(mat_filename, 'psq', 'params', 'pislquant');
    fprintf('Saved .mat file for FOV-shift workflow to: %s\n', mat_filename);

    %% Optionally, plot
    if nargin > 3 && flag_plot
        pge2.plot(psg, sys_ge);
    end

    
end