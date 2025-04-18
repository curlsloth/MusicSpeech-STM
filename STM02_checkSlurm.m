listing = dir('HPC_slurm/STM01/**/*.out');

% Specify the date and time range
start_date = datetime('2024-06-28 11:30:00');
end_date = datetime('2024-06-28 23:59:59');

for nOut = 1:length(listing)
    if (datetime(listing(nOut).date) > start_date) && (datetime(listing(nOut).date) < end_date)
        output_filename = [listing(nOut).folder,'/',listing(nOut).name];
        check_code = sum(contains(readlines(output_filename),'Successful STM!'));
        if ~check_code
            disp(['error here: ', output_filename])
        end
    end
end