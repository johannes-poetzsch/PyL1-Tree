function [skeletonBranches, skeletonPoints, tlsPoints] = L1MedianAlgorithm(seedPoints, tlsPoints, parameters)
    globalBar = waitbar(0,'L_1 Median: Start');    % The globalBar is used to show the step being execuated by the L1-Tree algorithm
    iterationNum = 0;
    skeletonBranches = [];
    skeletonPoints = seedPoints;
    maxIterationNum = ceil(log(parameters.baseRadius*2/parameters.searchRange)/log(parameters.iterationSpeed + 1))*parameters.maxIterationNumFixedRange;
    tlsPoints = L1MedianCalculateDensity(tlsPoints,parameters,globalBar,iterationNum,maxIterationNum);
    while parameters.searchRange <= parameters.baseRadius*2
        iterationNumFixedRange = 0;
        while iterationNumFixedRange < parameters.maxIterationNumFixedRange
            localBar = waitbar(0,'One iteration: start','Position',[100,100,270,60]);    % The localBar is used to show the information about the current iteration, including the search range, the iteration number, and the average moving distance
            skeletonPoints = L1MedianSkeletonNeighbors(skeletonPoints,tlsPoints,parameters,localBar);
            skeletonPoints = L1MedianCalculateSigma(skeletonPoints,parameters,localBar);
            skeletonPoints = L1MedianCalculateAverageTerm(skeletonPoints,parameters,localBar);
            skeletonPoints = L1MedianCalculateRepulsionTerm(skeletonPoints,parameters,localBar);
            [skeletonPoints,averageMovingDistance] = L1MedianUpdateSkeletonPoints(skeletonPoints,parameters,localBar);
            iterationNumFixedRange = iterationNumFixedRange + 1;
            iterationNum = iterationNum + 1;
            close(localBar);
            showInfo = ['L_1 Median: ',num2str(parameters.searchRange),' ',num2str(iterationNumFixedRange),' ',num2str(averageMovingDistance)];
            waitbar(iterationNum/maxIterationNum,globalBar,showInfo);
        end
        skeletonPoints = L1MedianRemoveSkeletonPoints(skeletonPoints,parameters,globalBar,iterationNum,maxIterationNum);
        skeletonPoints = L1MedianSmoothSigma(skeletonPoints,parameters,globalBar,iterationNum,maxIterationNum);
        [newBranches,skeletonPoints] = L1MedianSearchNewBranches(skeletonPoints,parameters,globalBar,iterationNum,maxIterationNum);
        skeletonBranches = [skeletonBranches,newBranches];
        parameters.searchRange = parameters.searchRange*(1 + parameters.iterationSpeed);
    end
    close(globalBar);
end
