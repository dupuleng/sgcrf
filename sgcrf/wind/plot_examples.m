
% Generate histograms for 4 examples
idx = [1 7 8 9];
for j=1:length(idx)
  i = idx(j);
  Yp = mvnrnd(repmat(P(s(i),:),10000,1), Sigma);
  Yp2_norm = mvnrnd(repmat(P2_norm(s(i),:),10000,1),Sigma2);
  Yp2 = inv_copula(Y(t,:), Yp2_norm);
  Yp3_norm = mvnrnd(repmat(P3_norm(s(i),:),10000,1),Sigma3);
  Yp3 = inv_copula(Y(t,:), Yp3_norm);

  [f xi] = ksdensity(sum(Yp,2));
  [f2 xi2] = ksdensity(sum(Yp2,2));
  [f3 xi3] = ksdensity(sum(Yp3,2));

  figure;
  hold on;
  plot(xi, f, 'b', 'LineWidth', 1);
  plot(xi2, f2, 'r', 'LineWidth', 1); 
  plot(xi3, f3, 'Color', [0 .4 0], 'LineWidth', 1);
  set(gca, 'FontSize', 6);
  axis([0 60 0 0.4]);
  if (j == 1)
    legend('GCRF', 'GCRF + copula', 'Sparse GCRF + copula', 'Location', 'NorthWest');
  end
    
  prepare_figure(sprintf('hist%d.pdf', j), [4 3], 'y', 'Prob(Energy = y)')  
end

% Generate sample graphs for a few different examples
[idx j] = find(Y(s,1:24) > 0.5);
idx = unique(idx);
for j = 1:5
  i = idx(j);
  samples = 10;
  Yp = mvnrnd(repmat(P(s(i),:),samples,1), Sigma);
  Yp2_norm = mvnrnd(repmat(P2_norm(s(i),:),samples,1),Sigma2);
  Yp2 = inv_copula(Y(t,:), Yp2_norm);
  Yp3_norm = mvnrnd(repmat(P3_norm(s(i),:),samples,1),Sigma3);
  Yp3 = inv_copula(Y(t,:), Yp3_norm);
  
  figure;
  hold on;
  ax1 = plot(Yp(:, 1:24)', 'Color', hex_color('ac d7 e6'));
  ax2 = plot(P(s(i),:), 'b', 'LineWidth', 1);
  ax3 = plot(Y(s(i), 1:24), 'k', 'LineWidth', 1);
  set(gca, 'FontSize', 6);
  xlim([1 24]);
  ylim([0 1]);
  legend([ax2 ax1(1) ax3], 'Mean', 'Sample', 'Truth', 'Location', 'NorthWest');

  figure;
  hold on;
  ax1 = plot(Yp2(:, 1:24)', 'Color', hex_color('f0 c0 ca'));
  ax2 = plot(P2(s(i),:), 'r', 'LineWidth', 1);
  ax3 = plot(Y(s(i), 1:24), 'k', 'LineWidth', 1);
  set(gca, 'FontSize', 6);
  xlim([1 24]);
  ylim([0 1]);
  
  figure;
  hold on;
  plot(Yp3(:, 1:24)', 'Color', hex_color('90 ed 90'));
  plot(P3(s(i),1:24), 'Color', hex_color('00 63 00'), 'LineWidth', 1);
  plot(Y(s(i), 1:24), 'k', 'LineWidth', 1);
  set(gca, 'FontSize', 6);
  xlim([1 24]);
end
figure(29); prepare_figure('gcrf_sample.pdf', [3.49 2.6], 'Hour', 'Energy');
figure(30); prepare_figure('gcrf_copula_sample.pdf', [3.49 2.6], 'Hour', 'Energy');
figure(31); prepare_figure('sgcrf_copula_sample.pdf', [3.49 2.6], 'Hour', 'Energy');

