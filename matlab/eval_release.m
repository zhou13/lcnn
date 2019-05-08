function eval_release(image_path, line_gt_path, output_file, result_path, output_size)

lineThresh = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999];

nLineThresh = size(lineThresh, 2);
sumtp = zeros(nLineThresh, 1);
sumfp = zeros(nLineThresh, 1);
sumgt = zeros(nLineThresh, 1);

listing = dir(image_path);
numResults = size(listing, 1);

for index=1:numResults
  filename = listing(index).name;
  if length(filename) == 1 || length(filename) == 2
    continue;
  end
  filename = filename(1:end-4);
  fprintf('processed %d/%d\n', index - 2, numResults - 2)
  gtname = [line_gt_path, '/', filename, '_line.mat'];
  imgname = [image_path, filename, '.jpg'];
  
  I = imread(imgname);
  height = size(I,1);
  width = size(I,2);
  
  % convert GT lines to binary map
  gtlines = load(gtname);
  gtlines = gtlines.lines;
  
  ne = size(gtlines,1);
  edgemap0 = zeros(height, width);
  for k = 1:ne
    x1 = gtlines(k,1);
    x2 = gtlines(k,3);
    y1 = gtlines(k,2);
    y2 = gtlines(k,4);
    
    vn = ceil(sqrt((x1-x2)^2+(y1-y2)^2));
    cur_edge = [linspace(y1,y2,vn).', linspace(x1,x2,vn).'];
    for j = 1:size(cur_edge,1)
      yy = round(cur_edge(j,1));
      xx = round(cur_edge(j,2));
      if yy <= 0
        yy = 1;
      end
      if xx <= 0
        xx = 1;
      end
      edgemap0(yy,xx) = 1;
    end
  end
  
  parfor m=1:nLineThresh
    resultname = [result_path, '/', num2str(lineThresh(m)), '/', sprintf('%06d', index - 3), '.mat'];
    resultlines = load(resultname);
    resultlines = resultlines.lines;
    ne = size(resultlines,1);
    edgemap1 = zeros(height, width);
    for k = 1:ne
      x1 = resultlines(k,2) * width / output_size;
      y1 = resultlines(k,1) * height / output_size;
      x2 = resultlines(k,4) * width / output_size;
      y2 = resultlines(k,3) * height / output_size;

      vn = ceil(sqrt((x1-x2)^2+(y1-y2)^2));
      cur_edge = [linspace(y1,y2,vn).', linspace(x1,x2,vn).'];
      for j = 1:size(cur_edge,1)
        yy = round(cur_edge(j,1) - 0.5);
        xx = round(cur_edge(j,2) - 0.5);
        if yy <= 0
          yy = 1;
        end
        if xx <= 0
          xx = 1;
        end
        if yy > height
          yy = height;
        end
        if xx > width
          xx = width;
        end
        edgemap1(yy,xx) = 1;
      end
    end
    
    [matchE1,matchG1] = correspondPixels(edgemap1,edgemap0,0.01);
    matchE = double(matchE1 > 0);

    sumtp(m, 1) = sumtp(m, 1) + sum(matchE(:));
    sumfp(m, 1) = sumfp(m, 1) + sum(edgemap1(:)) - sum(matchE(:));
    sumgt(m, 1) = sumgt(m, 1) + sum(edgemap0(:));
  end
end
save(output_file, 'sumtp', 'sumfp', 'sumgt');
end
