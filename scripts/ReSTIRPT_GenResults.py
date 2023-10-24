from falcor import *
import os
import time
import sys

gRenderReference = False
#gTestRealTime = True
gAntiAliased = False
gTestSceneName = "VeachAjar"
gTestSceneId = -1
gExperimentName = "main"
gDirectLighting = True

gAnimateScene = False

# "sweep through spp mode"
gTestSceneName="VeachAjar_StaticCamera"

if os.environ.get('RenderRef') != None:
    gRenderReference = True if int(os.environ['RenderRef']) == 1 else False
if os.environ.get('ExperimentName') != None:
    gExperimentName = os.environ['ExperimentName']
if os.environ.get('Scene') != None:
    gTestSceneName = os.environ['Scene']
if os.environ.get('Antialiasing') != None:
    gAntiAliased = True if int(os.environ['Antialiasing']) == 1 else False


if os.environ.get('UseDirectLighting') != None:
    gDirectLighting = True if int(os.environ['UseDirectLighting']) == 1 else False

def check_and_create_folder(folder):
    isdir = os.path.isdir(folder)
    if not isdir:
        os.mkdir(folder)

def render_graph_PathTracer(scenename):
    g = RenderGraph("PathTracer")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("PathTracer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("ImageLoader.dll")
    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1, 'maxSurfaceBounces': 9, 'maxDiffuseBounces': 9, 'maxSpecularBounces': 9, 'maxTransmissionBounces': 9, 'useRTXDI': gDirectLighting, 'disableDirectIllumination': not gDirectLighting, 'useReSTIRPT': True, 'emissiveSampler': EmissiveLightSamplerType.Power, 'useLambertianDiffuse': True})
    g.addPass(PathTracer, "PathTracer")
    VBufferRT = createPass("VBufferRT", {'samplePattern': SamplePattern.Center if not gAntiAliased else SamplePattern.Halton, 'sampleCount': 1 if not gAntiAliased else 32, 'useAlphaTest': True})
    g.addPass(VBufferRT, "VBufferRT")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': AccumulatePrecision.Double})
    g.addPass(AccumulatePass, "AccumulatePass")
    AccumulatePass2 = createPass("AccumulatePass", {'enabled': True, 'precisionMode': AccumulatePrecision.Double})
    g.addPass(AccumulatePass2, "AccumulatePass2")
    AccumulatePassSubpath = createPass("AccumulatePass", {'enabled': True, 'precisionMode': AccumulatePrecision.Double})
    g.addPass(AccumulatePassSubpath, "AccumulatePassSubpath")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    if False:#not gRenderReference:
        ImageLoader = createPass("ImageLoader", {"filename": "lut/" + scenename + "RefMono.exr", "srgb": False})
        g.addPass(ImageLoader, "ImageLoader")

    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer.mvec")
    if False:#not gRenderReference:
        g.addEdge("ImageLoader.dst", "PathTracer.integratedPHat")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("PathTracer.subColor", "AccumulatePassSubpath.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    #g.markOutput("PathTracer.krgb")
    g.addEdge("PathTracer.krgb", "AccumulatePass2.input")
    #g.markOutput("AccumulatePass2.output")
    g.markOutput("AccumulatePass.output")
    #g.markOutput("AccumulatePassSubpath.output")
    return g

def render_graph_PathTracerDLSSD():
    g = RenderGraph("PathTracer")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("PathTracer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1, 'maxSurfaceBounces': 9, 'maxDiffuseBounces': 9, 'maxSpecularBounces': 9, 'maxTransmissionBounces': 9, 'useRTXDI': gDirectLighting, 'disableDirectIllumination': not gDirectLighting, 'useReSTIRPT': True, 'emissiveSampler': EmissiveLightSamplerType.Power, 'useLambertianDiffuse': True})
    g.addPass(PathTracer, "PathTracer")
    GBufferRT = createPass("GBufferRT", {'samplePattern': SamplePattern.Halton, 'sampleCount': 32, 'useAlphaTest': True})
    g.addPass(GBufferRT, "GBufferRT")
    AccumulatePass = createPass("AccumulatePass", {'enabled': False, 'precisionMode': AccumulatePrecision.Single})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapperDLSSD = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapperDLSSD, "ToneMapperDLSSD")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    DLSSDPass = createPass("DLSSDPass", {'enabled': True, 'exposure' : 0.0})
    g.addPass(DLSSDPass, "DLSSDPass")

    g.addEdge("GBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("GBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("GBufferRT.mvec", "PathTracer.mvec")
    g.addEdge("PathTracer.color", "AccumulatePass.input")

    g.addEdge("GBufferRT.mvec",                                     "DLSSDPass.mvec")
    g.addEdge("GBufferRT.linearZ",                                  "DLSSDPass.depth")
    g.addEdge("GBufferRT.normWRoughnessMaterialID",                 "DLSSDPass.normalRoughness")
    g.addEdge("PathTracer.albedo",                          "DLSSDPass.diffuseAlbedo")
    g.addEdge("PathTracer.specularAlbedo",                  "DLSSDPass.specularAlbedo")
    # Optional guides for delta reflection / transmission
    g.addEdge("GBufferRT.posW",                                     "DLSSDPass.posW")
    g.addEdge("GBufferRT.mtlData",                                  "DLSSDPass.mtlData")
    g.addEdge("PathTracer.nrdDeltaReflectionReflectance",   "DLSSDPass.reflectionAlbedo")
    g.addEdge("PathTracer.nrdDeltaTransmissionReflectance", "DLSSDPass.transmissionAlbedo")

    g.addEdge("AccumulatePass.output", "DLSSDPass.color")
    g.addEdge("DLSSDPass.output", "ToneMapperDLSSD.src")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    g.markOutput("ToneMapperDLSSD.dst")
    g.markOutput("ToneMapper.dst")
    g.markOutput("DLSSDPass.output")
    g.markOutput("AccumulatePass.output")

    return g


# Load scene, animate camera, load methods

#sceneRootFolder = "E:/siggraph_gris_results/"
sceneRootFolder = os.environ['P4SCENES'].replace('\\','/') + "/"
camCaptureRootFolder = os.getcwd().replace('\\','/') + "/../../Docs/sig23-ReSTIR_Subpath/CamCaptures_Final/"
resultFolder = os.getcwd().replace('\\','/') + "/../../Docs/sig23-ReSTIR_Subpath/Results_Paper/"
check_and_create_folder(resultFolder)
if gRenderReference:
    resultFolder += "references/"
else:
    resultFolder += gExperimentName + "/"
check_and_create_folder(resultFolder)

warmupFrames = 64
# stillTimingFrames = 256
# offlineFrames = 256

# find scene name in list
# original microop

sceneName = ["VeachAjar_StaticCamera", "ZeroDay_StaticCamera", "LivingRoom_StaticCamera", "House_StaticCamera", "VeachAjar", "ZeroDay", "LivingRoom", "Classroom", "DiningRoom"]
scenePaths = {"VeachAjar": "VeachAjar/VeachAjar.pyscene", "VeachAjar_StaticCamera": "VeachAjar/VeachAjar.pyscene", 
"ZeroDay": "ZeroDay/MEASURE_SEVEN/MEASURE_SEVEN_ANIMATED.pyscene",
"ZeroDay_StaticCamera": "ZeroDay/MEASURE_SEVEN/MEASURE_SEVEN_ANIMATED.pyscene", 
"LivingRoom": "pbrt-v4-scenes/living-room/living-room.pyscene",
"LivingRoom_StaticCamera": "pbrt-v4-scenes/living-room/living-room.pyscene", "House_StaticCamera": "house/house.pyscene",
"Classroom": "pbrt-v4-scenes/classroom/classroom.pyscene", "DiningRoom": "pbrt-v4-scenes/dining-room/dining-room.pyscene"}

# scenePaths = ["VeachAjar/VeachAjar.pyscene", "Carousel/Carousel_FullScene_restirpt.pyscene", "AmusementPark/AmusementPark.pyscene",
#  "Carousel/Carousel_FullScene_restirpt.pyscene", "pbrt-v4-scenes/crown/crown_restirpt.pbrt", "Kitchen/Kitchen.pyscene", "pbrt-rendering-resources/kitchen/scene-v4.pbrt",
#  "Minecraft/lost-empire.pyscene", "ZeroDay/MEASURE_SEVEN/MEASURE_SEVEN_ANIMATED.pyscene"]

gTestOne = [sceneName.index(gTestSceneName)]

referenceFrames = {"VeachAjar": 100000, "VeachAjar_StaticCamera": 100000, "LivingRoom": 100000, "LivingRoom_StaticCamera":100000, 
"ZeroDay": 100000, "ZeroDay_StaticCamera": 100000, "House_StaticCamera": 1000000, "Classroom": 100000, "DiningRoom": 100000}

toneMappingCompensation = {"VeachAjar": 0.0, "VeachAjar_StaticCamera": 0.0, "LivingRoom": 0.0, "LivingRoom_StaticCamera":0.0, "ZeroDay": 0.0, 
"ZeroDay_StaticCamera": 0.0, "House_StaticCamera": 2.0, "Classroom": 0.0, "DiningRoom": 2.0}

# TODO: pick better camera trajectory and captured frame later
# TODO: make scene camera movement speed more uniform

# parse captured frame id from file name

gUseDynamicCamera = True
# if no suffix, it is treated as a static camera scene
gCaptureVideo = False


# if gExperimentName == "main" and not gRenderReference:
#     PathTracer = render_graph_PathTracerDLSSD()
# else:
PathTracer = render_graph_PathTracer(sceneName[gTestOne[0]])
try: m.addGraph(PathTracer)
except NameError: None

# compress hit Info? or should we?
m.setMogwaiUISceneBuilderFlag(buildFlags = SceneBuilderFlags.UseCompressedHitInfo)

m.resizeSwapChain(1920, 1080)

m.ui = False

# create result folder for specific experiment ID
check_and_create_folder(resultFolder)

def findFileInFolder(name, path):
    for root, dirs, files in os.walk(path):
        for f in files:
            if name + "_" in f:
                return f

for sceneId, scene in enumerate(sceneName):
    if sceneId not in gTestOne:
        continue

    m.loadScene(path=sceneRootFolder + scenePaths[scene])

    m.scene.camera.animated = False
    m.scene.animated = False
    m.scene.loopAnimations = False
    # m.clock.stop()

    # search 
    sceneCameraPath = findFileInFolder(scene, camCaptureRootFolder)
    if "StaticCamera" in sceneCameraPath:
        gUseDynamicCamera = False
        capturedFrameId = 0
    else:
        gUseDynamicCamera = True
        capturedFrameId = int(sceneCameraPath.split(".")[0].split("_")[-1])

    f = open(camCaptureRootFolder + sceneCameraPath, "r")
    lines = f.readlines()

    outputPosition = []
    outputTarget = []
    outputUp = []

    for i in range(len(lines)):
        if lines[i] == '\n':
            continue
        else:
            if i % 4 == 0:
                outputPosition.append(float3(float(lines[i].split(" ")[0]), float(lines[i].split(" ")[1]), float(lines[i].split(" ")[2].split("\\")[0])))
            elif i % 4 == 1:
                outputTarget.append(float3(float(lines[i].split(" ")[0]), float(lines[i].split(" ")[1]), float(lines[i].split(" ")[2].split("\\")[0])))
            elif i % 4 == 2:
                outputUp.append(float3(float(lines[i].split(" ")[0]), float(lines[i].split(" ")[1]), float(lines[i].split(" ")[2].split("\\")[0])))

    animationFrames = len(outputPosition)
    print("lenanim", animationFrames)

    # use default camera position
    if not gUseDynamicCamera and animationFrames == 0:
        outputPosition.append(m.scene.camera.position)
        outputTarget.append(m.scene.camera.target)
        outputUp.append(m.scene.camera.up)

    # scene-specific params
    #if gToneMap:
    PathTracer.updatePass("ToneMapper", {'autoExposure': False, 'exposureCompensation': toneMappingCompensation[scene]})
    # if gExperimentName == "main" and not gRenderReference:
    #     PathTracer.updatePass("ToneMapperDLSSD", {'autoExposure': False, 'exposureCompensation': toneMappingCompensation[scene]})


    kDirectOrNot = "" if gDirectLighting else "_Indirect" 

    # render reference
    # path tracing mode
    if gRenderReference:

        m.scene.camera.position = outputPosition[capturedFrameId]
        m.scene.camera.target = outputTarget[capturedFrameId]
        m.scene.camera.up = outputUp[capturedFrameId]

        m.frameCapture.outputDir = resultFolder
        check_and_create_folder(m.frameCapture.outputDir)

        commonOptionsNoReSTIRPT = {'useRTXDI': gDirectLighting, 'useReSTIRPT': False, 'emissiveSampler': EmissiveLightSamplerType.Power, 'useLambertianDiffuse': True, 'disableDirectIllumination': not gDirectLighting }
        PathTracer.updatePass("PathTracer", {'samplesPerPixel': 1, 'totalSamplesPerPixel': 1, 'useRussianRoulette': False, **commonOptionsNoReSTIRPT})

        PathTracer.updatePass("AccumulatePass", {'enableAccumulation': False, 'precisionMode': AccumulatePrecision.Double})
        m.clock.framerate = 30
        m.clock.play()

        for i in range(capturedFrameId+1):
            print(i)            
            m.scene.camera.position = outputPosition[i]
            m.scene.camera.target = outputTarget[i]
            m.scene.camera.up = outputUp[i]
            m.renderFrame()                
            
        m.clock.pause()

        PathTracer.updatePass("AccumulatePass", {'enableAccumulation': True, 'precisionMode': AccumulatePrecision.Double})
        for i in range(referenceFrames[scene]):
            m.renderFrame()
            if i % 100000 == 0:
                print(i)
                m.frameCapture.baseFilename = scene + ("" if gAntiAliased else "_aliased_") + kDirectOrNot + "_Reference=" + str(i) + "Frames"
                m.frameCapture.capture()
        
        m.frameCapture.baseFilename = scene + ("" if gAntiAliased else "_aliased_") + kDirectOrNot + "_Reference=" + str(referenceFrames[scene]) + "Frames"
        m.frameCapture.capture()
    else:
        sceneResultsFolder = scene + "_Results"

        m.frameCapture.outputDir = resultFolder + sceneResultsFolder
        check_and_create_folder(m.frameCapture.outputDir)

        # real time methods
        kOurs = 0
        kReSTIRPT = 1
        kOursReSTIRPT = 2
        kMMIS = 3
        kPT = 4

        methods = []

        # Fig 1: ours vs MMIS
        if gExperimentName == "suffixes":
            sppList = [1,2,3,4,5,6,7,8]
            # ours 1 to 8 supporting neighbors
            for spp in sppList:
                methods.append(["ours_" + str(spp) + "suffix", kOurs, {'ReSTIRPTOptions': ReSTIRPathTracingOptions(subpathSetting=SubpathSettings(finalGatherSuffixCount=spp))}])

            # MMIS 1 to 8 supporting neighbors
            for spp in sppList:
                methods.append(["mmis_" + str(spp) + "suffix", kMMIS, {'ReSTIRPTOptions': ReSTIRPathTracingOptions(subpathSetting=SubpathSettings(finalGatherSuffixCount=spp, useMMIS=True))}])
        elif gExperimentName == "prefixes":
            sppList = [2, 8, 32, 128]
            for spp in sppList:
                methods.append(["ours_" + str(spp) + "prefix", kOurs, {'ReSTIRPTOptions': ReSTIRPathTracingOptions(subpathSetting=SubpathSettings(numIntegrationPrefixes=spp))}])
            methods.append(["ours_" + str(spp) + "prefix_canonical", kOurs, {'ReSTIRPTOptions': ReSTIRPathTracingOptions(subpathSetting=SubpathSettings(numIntegrationPrefixes=spp, generateCanonicalSuffixForEachPrefix=True))}])
        elif gExperimentName == "main":
            methods.append(["ours", kOurs, {'ReSTIRPTOptions': ReSTIRPathTracingOptions(subpathSetting=SubpathSettings(numIntegrationPrefixes=128))}])
            methods.append(["mmis", kMMIS, {'ReSTIRPTOptions': ReSTIRPathTracingOptions(subpathSetting=SubpathSettings(numIntegrationPrefixes=64, finalGatherSuffixCount=3, useMMIS=True))}])
            methods.append(["restirpt", kReSTIRPT, {'samplesPerPixel': 1, 'ReSTIRPTOptions': ReSTIRPathTracingOptions(subpathReuse=False)}])
            methods.append(["pt", kPT, {'samplesPerPixel': 1}])
        elif gExperimentName == "converge" or gExperimentName == "convergeIndie":
            methods.append(["ours_canonical", kOurs, {'ReSTIRPTOptions': ReSTIRPathTracingOptions(subpathSetting=SubpathSettings(numIntegrationPrefixes=128, generateCanonicalSuffixForEachPrefix=True))}])
            methods.append(["ours", kOurs, {'ReSTIRPTOptions': ReSTIRPathTracingOptions(subpathSetting=SubpathSettings(numIntegrationPrefixes=128))}])
            methods.append(["mmis", kMMIS, {'ReSTIRPTOptions': ReSTIRPathTracingOptions(subpathSetting=SubpathSettings(numIntegrationPrefixes=64, finalGatherSuffixCount=3, useMMIS=True))}])
            methods.append(["restirpt", kReSTIRPT, {'samplesPerPixel': 1, 'ReSTIRPTOptions': ReSTIRPathTracingOptions(subpathReuse=False)}])
            methods.append(["pt", kPT, {'samplesPerPixel': 1}])

# run the program
#############

        # render some warm up frames to ensure the program is created
        m.renderFrame()
        m.renderFrame()
        m.renderFrame()


        for id in range(len(methods)):            

            PathTracer.updatePass("AccumulatePass", {'enableAccumulation': False, 'precisionMode': AccumulatePrecision.Double})
            methodName = methods[id][0]
            print("method " + str(id) + "/" + str(len(methods)))
            print("methodName " + methodName)

            PathTracer.updateDict("PathTracer", {**methods[id][2]})
            m.renderFrame()
            m.renderFrame()
            PathTracer.updateModeId("PathTracer", methods[id][1])
            m.renderFrame()
            m.renderFrame()
            
            # warm up
            if gUseDynamicCamera:
                m.scene.camera.position = outputPosition[capturedFrameId]
                m.scene.camera.target = outputTarget[capturedFrameId]       
                m.scene.camera.up = outputUp[capturedFrameId]
            ###########

            m.clock.framerate = 30
            m.clock.stop()

            if gExperimentName != "converge":
                m.scene.camera.position = outputPosition[0]
                m.scene.camera.target = outputTarget[0]
                m.scene.camera.up = outputUp[0]

            for i in range(warmupFrames):
                m.renderFrame()

            if gCaptureVideo:
                m.timingCapture.captureFrameTime(str(m.frameCapture.outputDir) + '/' + methodName + "_Animated.csv")
                m.clock.play()

                # capture 
                # capture timings (use this if we need to capture a video)
                for i in range(animationFrames if gUseDynamicCamera else 20):
                    if i == capturedFrameId + 1:
                        break

                if gUseDynamicCamera:
                    m.scene.camera.position = outputPosition[i]
                    m.scene.camera.target = outputTarget[i]
                    m.scene.camera.up = outputUp[i]
                m.renderFrame()
                                    
                m.timingCapture.captureFrameTime("")                
                m.clock.stop()

            # capture images            
            m.frameCapture.outputDir = resultFolder + sceneResultsFolder + "/" + methodName + "_FrameCapture"
            check_and_create_folder(m.frameCapture.outputDir)
            if not gCaptureVideo:
                m.timingCapture.captureFrameTime(str(m.frameCapture.outputDir) + '/' + "time.csv")

            # capture converged image
            if gExperimentName == "convergeIndie":
                # the dynamic way
                for i in range(256):
                    m.clock.play()
                    # capture at least 20 frame's timing
                    for j in range((capturedFrameId-19 if gUseDynamicCamera else 0), (capturedFrameId+1 if gUseDynamicCamera else 20)):
                    # for j in range(animationFrames if gUseDynamicCamera else 20):
                        if gUseDynamicCamera:
                            m.scene.camera.position = outputPosition[j]
                            m.scene.camera.target = outputTarget[j]
                            m.scene.camera.up = outputUp[j]
                        m.renderFrame()                
                        if gUseDynamicCamera and j == capturedFrameId or not gUseDynamicCamera and j == 19:
                            if not gCaptureVideo:
                                m.timingCapture.captureFrameTime("")                
                            m.frameCapture.baseFilename = "Frame" + str(i)
                            m.frameCapture.capture(True)      
                            break          
                    m.clock.stop()
            elif gExperimentName == "converge":
                # the static way
                PathTracer.updatePass("AccumulatePass", {'enableAccumulation': True, 'precisionMode': AccumulatePrecision.Double})
                for i in range(8192):
                    m.renderFrame()                
                    if i+1 in [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192]:
                        m.frameCapture.baseFilename = "Frame" + str(i)
                        m.frameCapture.capture(True)      
            else:
                m.clock.play()
                # capture at least 20 frame's timing
                for i in range(animationFrames if gUseDynamicCamera else 20):
                    if gUseDynamicCamera:
                        m.scene.camera.position = outputPosition[i]
                        m.scene.camera.target = outputTarget[i]
                        m.scene.camera.up = outputUp[i]
                    m.renderFrame()                
                    #m.captureScreen("Frame"+str(i), m.frameCapture.outputDir)    
                    if gUseDynamicCamera and i == capturedFrameId or not gUseDynamicCamera and i == 19:
                        if not gCaptureVideo:
                            m.timingCapture.captureFrameTime("")                
                        PathTracer.writeResults("PathTracer", str(m.frameCapture.outputDir))     
                        #m.frameCapture.baseFilename = "Frame" + str(i)
                        #m.frameCapture.capture(True)      
                        break          
                m.clock.stop()
                    
            m.frameCapture.outputDir = resultFolder + sceneResultsFolder #switch back

exit()
