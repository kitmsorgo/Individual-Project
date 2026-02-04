function make_directories(C)
% MAKE_DIRECTORIES
% Ensure data folders exist.

    dirs = {C.paths.data_root, C.paths.data_raw, C.paths.data_meta};
    for i = 1:numel(dirs)
        if ~exist(dirs{i}, 'dir')
            mkdir(dirs{i});
        end
    end
end
