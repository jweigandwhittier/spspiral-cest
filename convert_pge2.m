function convert_pge2(seq_filename, sys, cvs, pislquant, flag_plot)
% Converts a Pulseq file (.seq) to GE format (.pge)
%
%
% INPUTS:
%   seq_filename  - Path to the .seq file to be converted
%   sys           - Pulseq system limits object
%   cvs           - Structure containing GE CVs (e.g., cvs.xloc, cvs.yloc, cvs.zloc)
%   flag_plot     - Boolean (true/false) to plot the sequence
    
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
    psd_rf_wait = 148e-6; % Might need to grab the real value from the scanner?
    psd_grd_wait = 152e-6; 
    b1_max = 0.25; % [Gauss], should be high enough to handle SPSP pulses?
    g_max = sys.maxGrad/sys.gamma*100;
    slew_max = sys.maxSlew/sys.gamma/10;
    coil = 'hrmw'; % For Premier (UCSFMB3TMR)

    sys_ge = pge2.opts(psd_rf_wait, psd_grd_wait, b1_max, g_max, slew_max, coil);

    % Check sequence
    PNSwt = [1 1 1];   % Directional PNS weights, see pge2.pns()
    params = pge2.check(psg, sys_ge, 'PNSwt', PNSwt);

    % Validate versus Pulseq file
    seq = mr.Sequence();
    seq.read(seq_filename); % Turns out you actually have to load it like this??
    pge2.validate(psg, sys_ge, seq, [], 'row', [], 'plot', false);

    % Apply slice offset from CVs
    psg = pge2.translateFOVrf(psg, [cvs.xloc cvs.yloc cvs.zloc]);

    % Finally, convert
    out_filename = fullfile(filepath, [base_name '.pge']);
    pge2.serialize(psg, out_filename, 'pislquant', pislquant, 'params', params, 'checkHash', false);
    fprintf('Saved GE file to: %s\n', out_filename);

    %% Optionally, plot
    if nargin > 3 && flag_plot
        pge2.plot(psg, sys_ge);
    end

    
end