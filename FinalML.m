%% =========================
% FINAL RESEARCH PIPELINE
%% =========================
clc; clear; close all;

%% =========================
% 1. LOAD DATA
%% =========================
T = readtable("DatasetML.xlsx");
T = rmmissing(T);

% Remove text columns if present
T = removevars(T, intersect(T.Properties.VariableNames, ...
    ["Composition","WearConditions"]));

% Clean variable names
T.Properties.VariableNames = matlab.lang.makeValidName(T.Properties.VariableNames);

%% =========================
% 2. TARGET TRANSFORMATION
%% =========================
T.WearRate = log(T.WearRate);

Y = T.WearRate;
X = T(:, setdiff(T.Properties.VariableNames,"WearRate"));

%% =========================
% 3. TRAIN-TEST SPLIT
%% =========================
cv = cvpartition(height(T),'HoldOut',0.2);

Xtrain = X(training(cv),:);
Ytrain = Y(training(cv));

Xtest = X(test(cv),:);
Ytest = Y(test(cv));

%% =========================
% 4. ADVANCED MODEL (BOOSTING)
%% =========================

t = templateTree(...
    MaxNumSplits=60, ...
    MinLeafSize=5);

Mdl = fitrensemble(X, Y, ...
    'Method', 'LSBoost', ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct( ...
        'MaxObjectiveEvaluations', 25, ...
        'AcquisitionFunctionName', 'expected-improvement-plus'));

%% =========================
% 5. PREDICTION
%% =========================

Ytrain_pred = predict(Mdl,Xtrain);
Ytest_pred  = predict(Mdl,Xtest);

%% =========================
% 6. PERFORMANCE (BETTER METRICS)
%% =========================

RMSE_train = sqrt(mean((Ytrain - Ytrain_pred).^2));
RMSE_test  = sqrt(mean((Ytest  - Ytest_pred ).^2));

R2_train = 1 - sum((Ytrain - Ytrain_pred).^2) / sum((Ytrain - mean(Ytrain)).^2);
R2_test  = 1 - sum((Ytest  - Ytest_pred ).^2) / sum((Ytest  - mean(Ytest)).^2);

fprintf("Train RMSE = %.4f\n", RMSE_train);
fprintf("Test  RMSE = %.4f\n", RMSE_test);

fprintf("Train R² = %.4f\n", R2_train);
fprintf("Test  R² = %.4f\n", R2_test);

%% =========================
% 7. FEATURE IMPORTANCE
%% =========================
imp = predictorImportance(Mdl);
[impSorted, idx] = sort(imp,'descend');
names = string(Mdl.PredictorNames(idx));

figure
barh(impSorted,'FaceColor',[0.85 0.33 0.1])
yticklabels(names)
xlabel("Importance")
title("Feature Importance")
grid on

%% =========================
% PARITY PLOT (FIXED)
%% =========================
figure
scatter(Ytest, Ytest_pred, 60, Ytest_pred, 'filled')

colormap(turbo)
colorbar

hold on
plot([min(Ytest) max(Ytest)], [min(Ytest) max(Ytest)], ...
    'k--','LineWidth',2)

xlabel("Actual WearRate (log)",'FontWeight','bold')
ylabel("Predicted WearRate (log)",'FontWeight','bold')
title("Parity Plot (Boosting Model)",'FontWeight','bold')

grid on

%% =========================
% 9. CORRELATION HEATMAP
%% =========================
corrMat = corr(T{:,vartype('numeric')});

figure
imagesc(corrMat)
colormap(turbo)
colorbar

vars = T.Properties.VariableNames;
xticks(1:length(vars))
yticks(1:length(vars))

xticklabels(vars)
yticklabels(vars)

xtickangle(45)
title("Correlation Matrix")

%% =========================
% 10. PARTIAL DEPENDENCE (EXPLAINABILITY)
%% =========================
figure
plotPartialDependence(Mdl, names(1))
title("Effect of " + names(1))

%% =========================
%% =========================
% SIMPLE WORKING CONTOUR
%% =========================

figure('Color','w')

% Create grid
L_range = linspace(min(T.Load), max(T.Load), 25);
V_range = linspace(min(T.SlidingVelocity), max(T.SlidingVelocity), 25);

[Lg, Vg] = meshgrid(L_range, V_range);

% Base row (safe initialization)
X_base = Xtrain(1,:);

for k = 1:width(Xtrain)
    X_base{1,k} = mean(Xtrain{:,k});
end

% Prediction grid
Wear_pred = zeros(size(Lg));

for i = 1:numel(Lg)
    X_temp = X_base;

    X_temp.Load = Lg(i);
    X_temp.SlidingVelocity = Vg(i);

    Wear_pred(i) = predict(Mdl, X_temp);
end

%% 🔥 DIRECT PLOT (NO NORMALIZATION → NO ERROR)
contourf(Lg, Vg, Wear_pred, 20, 'LineColor','none')
colorbar

xlabel("Load (N)")
ylabel("Sliding Velocity (m/s)")
title("Wear Contour Map")

grid on
%% =========================
%% =========================
% 12. OPTIMIZATION (BEST CASE)
%% =========================
[~, idx_best] = min(Ytest_pred);   % ← fixed: was Y_pred
best_sample = Xtest(idx_best,:);

disp("Best Condition & Composition:")
disp(best_sample)
disp(best_sample)

figure('Color','w')

% Scatter with color mapping
scatter(T.Hardness, T.WearRate, 60, T.WearRate, ...
    'filled','MarkerEdgeColor','k')

colormap(turbo)
cb = colorbar;
cb.Label.String = 'log(Wear Rate)';

hold on

% Smooth trend line (polynomial fit)
p = polyfit(T.Hardness, T.WearRate, 2);
x_fit = linspace(min(T.Hardness), max(T.Hardness), 100);
y_fit = polyval(p, x_fit);

plot(x_fit, y_fit, 'r-', 'LineWidth', 3)

% Labels & styling
xlabel("Hardness (HV)",'FontSize',12,'FontWeight','bold')
ylabel("log(Wear Rate)",'FontSize',12,'FontWeight','bold')

title("Effect of Hardness on Wear Behavior", ...
    'FontSize',14,'FontWeight','bold')

grid on
set(gca,'FontSize',12)



figure('Color','w')

scatter(T.Hardness, T.WearRate, 60, T.Load, ...
    'filled','MarkerEdgeColor','k')

colormap(turbo)
cb = colorbar;
cb.Label.String = 'Load (N)';

hold on

p = polyfit(T.Hardness, T.WearRate, 2);
x_fit = linspace(min(T.Hardness), max(T.Hardness), 100);
y_fit = polyval(p, x_fit);

plot(x_fit, y_fit, 'r-', 'LineWidth', 3)

xlabel("Hardness (HV)",'FontWeight','bold')
ylabel("log(Wear Rate)",'FontWeight','bold')
title("Effect of Hardness on Wear",'FontWeight','bold')

grid on

%% =========================
% REAL OPTIMIZATION (FINAL CLEAN)
%% =========================

nSamples = 3000;

vars = Mdl.PredictorNames;

% Create new dataset
Xnew = array2table(zeros(nSamples, length(vars)), ...
    'VariableNames', vars);

for i = 1:length(vars)
    v = vars{i};
    Xnew.(v) = min(T.(v)) + (max(T.(v)) - min(T.(v))) .* rand(nSamples,1);
end

%% Normalize composition
elements = ["Ag","Al","C","Co","Cr","Cu","Fe","Mn","Mo","Nb","Ni","Si","Sn","Ti","V","W"];
validElems = intersect(elements, vars);

if ~isempty(validElems)
    compMat = Xnew{:, validElems};
    sumComp = sum(compMat,2);
    sumComp(sumComp==0) = 1;

    for j = 1:length(validElems)
        Xnew.(validElems(j)) = compMat(:,j) ./ sumComp * 100;
    end
end

%% Predict wear
Wear_pred_opt = predict(Mdl, Xnew);

%% Find best design
[minWear, idx_best] = min(Wear_pred_opt);

BestDesign = Xnew(idx_best,:);

%% Display
disp("🔥 OPTIMIZED RESULT:")
disp(BestDesign)
disp("Predicted Wear (log): " + minWear)

%% =========================
% SHAP ANALYSIS (KernelSHAP)
%% =========================

nExplain  = 50;       % samples to explain
nBg       = 100;      % background samples for expectation
nFeatures = width(Xtest);
varNames  = Mdl.PredictorNames;

% Background baseline (mean prediction reference)
bgIdx = randperm(height(Xtrain), nBg);
Xbg   = Xtrain(bgIdx, :);
f_bg  = mean(predict(Mdl, Xbg));   % E[f(x)]

% Subset of test samples to explain
explainIdx = randperm(height(Xtest), min(nExplain, height(Xtest)));
Xexplain   = Xtest(explainIdx, :);
Yexplain   = Ytest_pred(explainIdx);

shapValues = zeros(size(Xexplain,1), nFeatures);

for s = 1:size(Xexplain,1)
    x_s = Xexplain(s,:);

    for f = 1:nFeatures
        % With feature f  → use actual value
        X_with              = Xbg;
        X_with.(varNames{f}) = repmat(x_s.(varNames{f}), nBg, 1);
        f_with              = mean(predict(Mdl, X_with));

        % Without feature f → use background (marginal)
        f_without = f_bg;

        shapValues(s, f) = f_with - f_without;
    end
end

%% --- SHAP Summary Plot (Beeswarm-style) ---
figure('Color','w','Position',[100 100 750 500])

meanAbsSHAP = mean(abs(shapValues), 1);
[~, sortIdx] = sort(meanAbsSHAP, 'descend');

shapSorted  = shapValues(:, sortIdx);
namesSorted = string(varNames(sortIdx));

hold on
cmap = turbo(size(shapSorted,1));

for f = 1:nFeatures
    vals     = shapSorted(:, f);
    jitter   = (rand(size(vals)) - 0.5) * 0.35;

    % Color by feature value magnitude
    featVals = Xexplain{:, sortIdx(f)};
    normVals = (featVals - min(featVals)) / (max(featVals) - min(featVals) + 1e-8);

    scatter(vals, f + jitter, 40, normVals, 'filled', ...
        'MarkerEdgeColor','none','MarkerFaceAlpha',0.75)
end

colormap(turbo)
cb = colorbar;
cb.Label.String = 'Feature Value (low → high)';

yticklabels(namesSorted)
yticks(1:nFeatures)
xlabel('SHAP Value  (impact on log Wear Rate)','FontWeight','bold')
title('SHAP Summary Plot','FontWeight','bold','FontSize',13)
xline(0,'k--','LineWidth',1.2)
grid on
set(gca,'FontSize',11)

%% --- SHAP Bar Plot (Mean |SHAP|) ---
figure('Color','w','Position',[100 100 650 450])

barh(meanAbsSHAP(sortIdx), 'FaceColor',[0.2 0.6 0.9], ...
    'EdgeColor','none')
yticklabels(namesSorted)
yticks(1:nFeatures)
xlabel('Mean |SHAP Value|','FontWeight','bold')
title('Global Feature Importance (SHAP)','FontWeight','bold','FontSize',13)
grid on
set(gca,'FontSize',11)

%% --- SHAP Dependence Plot (Top Feature) ---
topFeat = namesSorted(1);

figure('Color','w','Position',[100 100 600 420])

scatter(Xexplain{:, topFeat}, shapSorted(:,1), 60, ...
    shapSorted(:,1), 'filled','MarkerEdgeColor','k','LineWidth',0.4)

colormap(turbo); colorbar
xlabel(topFeat,'FontWeight','bold','FontSize',12)
ylabel('SHAP Value','FontWeight','bold','FontSize',12)
title("SHAP Dependence Plot: " + topFeat,'FontWeight','bold','FontSize',13)
xline(mean(Xexplain{:,topFeat}),'r--','LineWidth',1.5)
grid on
set(gca,'FontSize',11)

fprintf("\nSHAP Analysis Complete. Top 3 influential features:\n")
for k = 1:min(3,nFeatures)
    fprintf("  %d. %s  (Mean |SHAP| = %.4f)\n", k, namesSorted(k), meanAbsSHAP(sortIdx(k)))
end
%% CONSTRAINED OPTIMIZATION (REALISTIC)

Xnew.SlidingDistance = max(min(Xnew.SlidingDistance, 5000), 10);
Xnew.Load = max(min(Xnew.Load, 50), 5);
Xnew.SlidingVelocity = max(min(Xnew.SlidingVelocity, 2), 0.01);

Wear_pred_opt2 = predict(Mdl, Xnew);

[minWear2, idx_best2] = min(Wear_pred_opt2);
BestDesign2 = Xnew(idx_best2,:);

disp("REALISTIC OPTIMUM:")
disp(BestDesign2)
disp("Predicted Wear (log): " + minWear2)

% Co-Cr-Ni OPTIMIZATION — NO SECTION BREAKS INSIDE
clc
disp("Starting optimization...")

% Verify
if ~exist('Mdl','var'), error('Mdl not found.'); end
if ~exist('T','var'),   error('T not found.');   end
disp("Dependencies OK")

% Settings
nSamples = 5000;
vars     = Mdl.PredictorNames;

% Step 1: Build Xsys
dataMatrix = zeros(nSamples, length(vars));
for k = 1:length(vars)
    dataMatrix(:,k) = mean(T.(vars{k}));
end
Xsys_opt = array2table(dataMatrix, 'VariableNames', vars);
disp("Xsys_opt initialized")

% Step 2: Find columns
findCol = @(tag) vars(cellfun(@(x) strcmpi(x,tag) || strcmpi(x,[tag,'_']), vars));
coCol   = findCol('Co');
crCol   = findCol('Cr');
niCol   = findCol('Ni');
disp("Columns found")

% Step 3: Compositions
Co_r    = rand(nSamples,1);
Cr_r    = rand(nSamples,1);
Ni_r    = rand(nSamples,1);
sumComp = Co_r + Cr_r + Ni_r;
if ~isempty(coCol), Xsys_opt.(coCol{1}) = (Co_r./sumComp)*100; end
if ~isempty(crCol), Xsys_opt.(crCol{1}) = (Cr_r./sumComp)*100; end
if ~isempty(niCol), Xsys_opt.(niCol{1}) = (Ni_r./sumComp)*100; end
disp("Compositions assigned")

% Step 4: Zero other elements
otherElems = {'Ag','Al','C','Cu','Fe','Mn','Mo','Nb','Si','Sn','Ti','V','W'};
for i = 1:length(otherElems)
    eCol = findCol(otherElems{i});
    if ~isempty(eCol)
        Xsys_opt.(eCol{1}) = zeros(nSamples,1);
    end
end
disp("Other elements zeroed")

% Step 5: Operating conditions (randomized within dataset range)
compColsUsed = [coCol, crCol, niCol];
for i = 1:length(otherElems)
    eCol = findCol(otherElems{i});
    if ~isempty(eCol), compColsUsed{end+1} = eCol{1}; end
end

for i = 1:length(vars)
    v = vars{i};
    % Skip composition AND hardness — hardness handled separately below
    if ~ismember(v, compColsUsed) && ~any(contains({v}, 'Hardness', 'IgnoreCase', true))
        vMin = min(T.(v));
        vMax = max(T.(v));
        Xsys_opt.(v) = vMin + (vMax - vMin) .* rand(nSamples,1);
    end
end
disp("Conditions randomized")

% Step 6: Estimate hardness from composition using dataset regression
%         then apply realistic constraint (50–550 HV)
hardCol = vars(contains(vars,'Hardness','IgnoreCase',true));

if ~isempty(hardCol)
    hName = hardCol{1};

    % Build composition matrix from dataset
    compData = [];
    compCols_present = {};

    for cc = [coCol, crCol, niCol]
        if ~isempty(cc) && ismember(cc{1}, T.Properties.VariableNames)
            compData         = [compData, T.(cc{1})];
            compCols_present{end+1} = cc{1};
        end
    end

    % Fit linear model: Hardness ~ Co + Cr + Ni
    if ~isempty(compData) && size(compData,2) >= 2
        mdl_hard = fitlm(compData, T.(hName));

        % Predict hardness for new compositions
        newCompData = zeros(nSamples, length(compCols_present));
        for cc = 1:length(compCols_present)
            newCompData(:,cc) = Xsys_opt.(compCols_present{cc});
        end

        H_pred = predict(mdl_hard, newCompData);

        % Add realistic scatter (±10% of predicted)
        noise  = 0.10 .* H_pred .* (2*rand(nSamples,1) - 1);
        H_pred = H_pred + noise;

        % Clamp to realistic range
        H_pred = max(min(H_pred, 550), 50);

        Xsys_opt.(hName) = H_pred;

        fprintf("Hardness estimated from composition (min=%.1f, max=%.1f, mean=%.1f HV)\n", ...
            min(H_pred), max(H_pred), mean(H_pred))
    else
        % Fallback: random within range if not enough composition columns
        vMin = min(T.(hName));
        vMax = min(max(T.(hName)), 550);
        Xsys_opt.(hName) = vMin + (vMax - vMin) .* rand(nSamples,1);
        fprintf("Hardness randomized (fallback): min=%.1f max=%.1f\n", vMin, vMax)
    end
else
    fprintf("Hardness column not found — skipping\n")
end

% Remaining constraints
loadCol = vars(contains(vars,'Load',            'IgnoreCase',true));
velCol  = vars(contains(vars,'SlidingVelocity', 'IgnoreCase',true));
distCol = vars(contains(vars,'SlidingDistance', 'IgnoreCase',true));

if ~isempty(loadCol), Xsys_opt.(loadCol{1}) = max(min(Xsys_opt.(loadCol{1}), 50),   5);    end
if ~isempty(velCol),  Xsys_opt.(velCol{1})  = max(min(Xsys_opt.(velCol{1}),   2),   0.01); end
if ~isempty(distCol), Xsys_opt.(distCol{1}) = max(min(Xsys_opt.(distCol{1}),  5000), 10);  end

disp("Constraints applied")

% Step 7: Align
Xsys_opt = Xsys_opt(:, vars);
disp("Columns aligned")

% Step 8: Predict
disp("Running predict...")
W = predict(Mdl, Xsys_opt);
W = double(W(:));
fprintf("Predictions: total=%d  valid=%d  NaN=%d\n", ...
    numel(W), sum(isfinite(W)), sum(isnan(W)))

% Step 9: Clean
W_clean  = W(isfinite(W));
X_clean  = Xsys_opt(isfinite(W), :);
if isempty(W_clean), error("All NaN. Column mismatch."); end
disp("Data cleaned")

% Step 10: Top 5 diverse compositions
nTop         = 5;
diversityGap = 10.0;   % increased from 5 to 10 at.% for more diversity

[W_sorted, W_order] = sort(W_clean, 'ascend');
X_sorted            = X_clean(W_order, :);

selectedIdx  = {};
selectedRows = {};

for i = 1:height(X_sorted)
    candidate = X_sorted(i,:);
    isDiverse = true;

    for j = 1:length(selectedRows)
        prev = selectedRows{j};

        dCo = 0; dCr = 0; dNi = 0;
        if ~isempty(coCol), dCo = abs(candidate.(coCol{1}) - prev.(coCol{1})); end
        if ~isempty(crCol), dCr = abs(candidate.(crCol{1}) - prev.(crCol{1})); end
        if ~isempty(niCol), dNi = abs(candidate.(niCol{1}) - prev.(niCol{1})); end

        % Also check hardness diversity
        dH = 0;
        if ~isempty(hardCol)
            dH = abs(candidate.(hardCol{1}) - prev.(hardCol{1}));
        end

        compDist = dCo + dCr + dNi;

        % Both composition AND hardness must differ
        if compDist < diversityGap || dH < 10
            isDiverse = false;
            break
        end
    end

    if isDiverse
        selectedIdx{end+1}  = i;
        selectedRows{end+1} = candidate;
    end

    if length(selectedRows) == nTop
        break
    end
end

fprintf("Found %d diverse optimal compositions\n", length(selectedRows))

% Step 11: Extract values for all 5 into arrays
coVals   = zeros(nTop,1);
crVals   = zeros(nTop,1);
niVals   = zeros(nTop,1);
hVals    = zeros(nTop,1);
loadVals = zeros(nTop,1);
velVals  = zeros(nTop,1);
distVals = zeros(nTop,1);
wLogVals = zeros(nTop,1);
wExpVals = zeros(nTop,1);

for r = 1:length(selectedRows)
    BestRow      = selectedRows{r};
    wLogVals(r)  = W_sorted(selectedIdx{r});
    wExpVals(r)  = exp(wLogVals(r));

    if ~isempty(coCol),   coVals(r)   = BestRow.(coCol{1});   end
    if ~isempty(crCol),   crVals(r)   = BestRow.(crCol{1});   end
    if ~isempty(niCol),   niVals(r)   = BestRow.(niCol{1});   end
    if ~isempty(hardCol), hVals(r)    = BestRow.(hardCol{1}); end
    if ~isempty(loadCol), loadVals(r) = BestRow.(loadCol{1}); end
    if ~isempty(velCol),  velVals(r)  = BestRow.(velCol{1});  end
    if ~isempty(distCol), distVals(r) = BestRow.(distCol{1}); end
end

% Step 12: Detailed display for each alloy
fprintf("\n")
for r = 1:length(selectedRows)
    fprintf("========================================\n")
    fprintf("  RANK %d — OPTIMAL Co-Cr-Ni DESIGN     \n", r)
    fprintf("========================================\n")
    fprintf("  log(Wear Rate) = %.4f\n",  wLogVals(r))
    fprintf("  Wear Rate      = %.6e\n",  wExpVals(r))
    fprintf("----------------------------------------\n")
    fprintf("  COMPOSITION (at.%%)\n")
    fprintf("  Co = %.2f  |  Cr = %.2f  |  Ni = %.2f\n", ...
        coVals(r), crVals(r), niVals(r))
    fprintf("----------------------------------------\n")
    fprintf("  PROPERTIES & CONDITIONS\n")
    fprintf("  Hardness         = %.2f HV\n",  hVals(r))
    fprintf("  Load             = %.2f N\n",   loadVals(r))
    fprintf("  Sliding Velocity = %.4f m/s\n", velVals(r))
    fprintf("  Sliding Distance = %.2f m\n",   distVals(r))
    fprintf("========================================\n\n")
end

% Step 13: Full comparison table
fprintf("\n")
fprintf("==========================================================================================\n")
fprintf("            COMPARISON TABLE — TOP %d Co-Cr-Ni OPTIMAL COMPOSITIONS                     \n", nTop)
fprintf("==========================================================================================\n")
fprintf("%-6s %-8s %-8s %-8s %-10s %-8s %-8s %-10s %-12s %-12s\n", ...
    "Rank","Co","Cr","Ni","Hard(HV)","Load","Vel","Dist","log(Wear)","Wear Rate")
fprintf("------------------------------------------------------------------------------------------\n")

for r = 1:length(selectedRows)
    fprintf("%-6d %-8.2f %-8.2f %-8.2f %-10.2f %-8.2f %-8.4f %-10.2f %-12.4f %-12.4e\n", ...
        r, coVals(r), crVals(r), niVals(r), hVals(r), ...
        loadVals(r), velVals(r), distVals(r), wLogVals(r), wExpVals(r))
end
fprintf("==========================================================================================\n")

% Step 14: Bar chart comparison of key properties
figure('Color','w','Position',[100 100 1100 420])

ranks  = 1:length(selectedRows);
labels = arrayfun(@(x) sprintf("Rank %d",x), ranks, 'UniformOutput', false);

% Composition subplot
subplot(1,4,1)
b1 = bar(ranks, [coVals, crVals, niVals], 'grouped');
b1(1).FaceColor = [0.2  0.6  0.9];
b1(2).FaceColor = [0.85 0.33 0.1];
b1(3).FaceColor = [0.47 0.67 0.19];
b1(1).EdgeColor = 'none';
b1(2).EdgeColor = 'none';
b1(3).EdgeColor = 'none';
xticks(ranks); xticklabels(labels); xtickangle(30)
ylabel("at.%",'FontWeight','bold')
title("Composition",'FontWeight','bold')
legend({"Co","Cr","Ni"},'Location','best','FontSize',8)
grid on; box off

% Hardness subplot
subplot(1,4,2)
b2 = bar(ranks, hVals, 'FaceColor','flat');
for r = 1:length(selectedRows)
    b2.CData(r,:) = [0.6 0.2 0.8];
end
b2.EdgeColor = 'none';
xticks(ranks); xticklabels(labels); xtickangle(30)
ylabel("Hardness (HV)",'FontWeight','bold')
title("Hardness",'FontWeight','bold')
ylim([0 600])
yline(550,'r--','LineWidth',1.5)
grid on; box off

% Wear rate subplot
subplot(1,4,3)
b3 = bar(ranks, wExpVals, 'FaceColor','flat');
for r = 1:length(selectedRows)
    b3.CData(r,:) = [0.85 0.33 0.1];
end
b3.EdgeColor = 'none';
xticks(ranks); xticklabels(labels); xtickangle(30)
ylabel("Wear Rate",'FontWeight','bold')
title("Wear Rate",'FontWeight','bold')
grid on; box off

% Conditions subplot
subplot(1,4,4)
b4 = bar(ranks, [loadVals, velVals*10, distVals/100], 'grouped');
b4(1).FaceColor = [0.9  0.6  0.1];
b4(2).FaceColor = [0.1  0.7  0.7];
b4(3).FaceColor = [0.8  0.2  0.2];
b4(1).EdgeColor = 'none';
b4(2).EdgeColor = 'none';
b4(3).EdgeColor = 'none';
xticks(ranks); xticklabels(labels); xtickangle(30)
title("Conditions (scaled)",'FontWeight','bold')
legend({"Load(N)","Vel","Dist"},'Location','best','FontSize',8)
grid on; box off

sgtitle("Top 5 Co-Cr-Ni Optimal Alloy Designs — Full Comparison", ...
    'FontSize',14,'FontWeight','bold')

% Step 15: Design space scatter with top 5 highlighted
figure('Color','w','Position',[100 100 700 550])

coAll = zeros(height(X_clean),1);
crAll = zeros(height(X_clean),1);
if ~isempty(coCol), coAll = X_clean{:,coCol{1}}; end
if ~isempty(crCol), crAll = X_clean{:,crCol{1}}; end

scatter(coAll, crAll, 18, W_clean, 'filled','MarkerFaceAlpha',0.35)
colormap(turbo)
cb = colorbar; cb.Label.String = 'log(Wear Rate)';
hold on

colors5 = lines(nTop);
for r = 1:length(selectedRows)
    scatter(coVals(r), crVals(r), 180, colors5(r,:), 'filled', ...
        'MarkerEdgeColor','k','LineWidth',1.5)
    text(coVals(r)+0.5, crVals(r)+0.5, ...
        sprintf("R%d\n%.0fHV", r, hVals(r)), ...
        'FontWeight','bold','FontSize',9,'Color',colors5(r,:))
end

xlabel("Co (at.%)",'FontWeight','bold','FontSize',12)
ylabel("Cr (at.%)",'FontWeight','bold','FontSize',12)
title("Co-Cr-Ni Design Space — Top 5 with Hardness Labels", ...
    'FontWeight','bold','FontSize',13)
grid on; box off