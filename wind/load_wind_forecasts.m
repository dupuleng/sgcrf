for i=1:7,
  wf{i} = csvread(['../../data/wind/windforecasts_wf' num2str(i) '.csv'], 1, 0);
  wf{i}(wf{i} == -10000) = nan;
  wf{i}(:,1) = date_to_timestamp(wf{i}(:,1));
  
  % compute most recent prediction
  t0 = wf{i}(:,1) + wf{i}(:,2)/24;
  [t0,idx] = unique(t0,'last');
  wf0{i} = [t0 wf{i}(idx,3:end)];
end
