function cell2csv(fname, C)
% CELL2CSV
% Very simple CSV writer for cell arrays of scalars/strings.

    fid = fopen(fname,'w');
    if fid == -1
        error('cell2csv:FileOpenError', 'Cannot open %s for writing.', fname);
    end

    cleaner = @(x) string(x);  % convert to string safely

    for i = 1:size(C,1)
        row = C(i,:);
        row_str = cleaner(row);
        line = strjoin(row_str, ',');
        fprintf(fid, '%s\n', line);
    end

    fclose(fid);
end
