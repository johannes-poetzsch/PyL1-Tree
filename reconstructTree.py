import matlab.engine

def reconstruct_tree():
    eng = matlab.engine.start_matlab()

#    parameters = eng.wrap("L1TreeInitialization", nargout=1)
#    fileName = eng.Wrapper('Tree_1')
#
#    tlsPoints, paths, parameters = eng.wrap("L1TreeExtractPaths", fileName, parameters, nargout=3)
#    seedPoints, tlsPoints = eng.wrap("L1TreeExtractSeedPoints",tlsPoints,paths,parameters, nargout=2)
#    
#    skeletonBranches, skeletonPoints, tlsPoints = eng.wrap("L1MedianAlgorithm", seedPoints, tlsPoints, parameters, nargout=3)
#
    globalBar = eng.eval("waitbar(0,'MST optimize: Start')")
#
#    treeModel, tlsPoints = eng.wrap("L1TreeConnect", skeletonBranches,skeletonPoints,tlsPoints,parameters,globalBar, nargout=2)
#    treeModel, tlsPoints = eng.wrap("L1TreePruneBranches", treeModel,tlsPoints,parameters,globalBar, nargout=2)
    eng.load("L1TreeAdjustBifurcation_Input.mat", nargout=0)
    treeModel = eng.eval("Wrapper(treeModel)")
    tlsPoints = eng.eval("Wrapper(tlsPoints)")
    parameters = eng.eval("Wrapper(parameters)")
    treeModel, tlsPoints = eng.wrap("L1TreeAdjustBifurcation", treeModel,tlsPoints,parameters,globalBar, nargout=2)
#    treeModel, tlsPoints = eng.wrap("L1TreeSmoothBranches", treeModel,tlsPoints,parameters,globalBar, nargout=2)
#    treeModel, tlsPoints = eng.wrap("L1TreeExtendSkeleton", treeModel,tlsPoints,parameters,globalBar, nargout=2)
#    treeModelCentrifugal, treeModelGaaliche = eng.wrap("L1TreeConstructModel", fileName,treeModel,tlsPoints,globalBar, nargout=2)

    #drawFileName = eng.L1TreeExportModel(fileName,treeModelCentrifugal)
    
if __name__ == '__main__':
    reconstruct_tree()