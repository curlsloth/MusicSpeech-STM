

listing = dir('HPC_slurm/**/*.out');
for nOut = 1:length(listing)
    output_filename = [listing(nOut).folder,'/',listing(nOut).name];
    check_code = sum(contains(readlines(output_filename),'Successful STM!'));
    if ~check_code
        disp(['error here: ', output_filename])
    end
end