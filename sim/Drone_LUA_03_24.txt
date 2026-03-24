sim=require'sim'

function sysCall_init() 
    particlesAreVisible=true
    simulateParticles=true
    fakeShadow=true
    
    particleCountPerSecond=430
    particleSize=0.005
    particleDensity=8500
    particleScatteringAngle=30
    particleLifeTime=0.5
    maxParticleCount=50

    -- Keep target only for vertical control:
    targetObj=sim.getObject('../target')
    sim.setObjectParent(targetObj,-1,true)

    d=sim.getObject('../base')
    heli=sim.getObject('..')

    propellerHandles={}
    jointHandles={}
    particleObjects={-1,-1,-1,-1}

    local ttype=sim.particle_roughspheres
               + sim.particle_cyclic
               + sim.particle_respondable1to4
               + sim.particle_respondable5to8
               + sim.particle_ignoresgravity

    if not particlesAreVisible then
        ttype=ttype+sim.particle_invisible
    end

    for i=1,4,1 do
        propellerHandles[i]=sim.getObject('../propeller['..(i-1)..']/respondable')
        jointHandles[i]=sim.getObject('../propeller['..(i-1)..']/joint')
        if simulateParticles then
            particleObjects[i]=sim.addParticleObject(
                ttype,
                particleSize,
                particleDensity,
                {2,1,0.2,3,0.4},
                particleLifeTime,
                maxParticleCount,
                {0.3,0.7,1}
            )
        end
    end

    -- Vertical control parameters (unchanged):
    pParam=2
    iParam=0
    dParam=0
    vParam=-2

    cumul=0
    lastE=0

    -- Horizontal control memory:
    pAlphaE=0
    pBetaE=0

    -- External command signals from Python:
    sigRoll='cmd_roll'
    sigPitch='cmd_pitch'
    sigYaw='cmd_yaw'
    sigThrust='cmd_thrust'

    cmdRoll=0
    cmdPitch=0
    cmdYaw=0
    cmdThrust=0

    if (fakeShadow) then
        shadowCont=sim.addDrawingObject(
            sim.drawing_discpts
            + sim.drawing_cyclic
            + sim.drawing_25percenttransparency
            + sim.drawing_50percenttransparency
            + sim.drawing_itemsizes,
            0.2,0,-1,1
        )
    end
end

function sysCall_cleanup() 
    if shadowCont then
        sim.removeDrawingObject(shadowCont)
    end
    for i=1,#particleObjects,1 do
        sim.removeParticleObject(particleObjects[i])
    end
end 

function sysCall_actuation() 
    local pos=sim.getObjectPosition(d)

    if (fakeShadow) then
        local itemData={pos[1],pos[2],0.002,0,0,0,1,0.2}
        sim.addDrawingObjectItem(shadowCont,itemData)
    end

    -- Read external commands from Python:
    local r=sim.getFloatSignal(sigRoll)
    local p=sim.getFloatSignal(sigPitch)
    local y=sim.getFloatSignal(sigYaw)
    local tCmd=sim.getFloatSignal(sigThrust)

    if r~=nil then cmdRoll=r end
    if p~=nil then cmdPitch=p end
    if y~=nil then cmdYaw=y end
    if tCmd~=nil then cmdThrust=tCmd end
    
    -------------------------------------------------
    -- Vertical control (unchanged)
    -------------------------------------------------
    local targetPos=sim.getObjectPosition(targetObj)
    pos=sim.getObjectPosition(d)
    local l=sim.getVelocity(heli)

    local e=(targetPos[3]-pos[3])
    cumul=cumul+e
    local pv=pParam*e
    local thrust=5.45+pv+iParam*cumul+dParam*(e-lastE)+l[3]*vParam + cmdThrust
    lastE=e
    
    -------------------------------------------------
    -- Horizontal control
    -- Replace target-position tracking with roll/pitch commands
    -------------------------------------------------
    local m=sim.getObjectMatrix(d)

    local vx={1,0,0}
    vx=sim.multiplyVector(m,vx)

    local vy={0,1,0}
    vy=sim.multiplyVector(m,vy)

    -- Roll channel:
    local alphaE=(vy[3]-m[12]) - cmdRoll
    local alphaCorr=0.25*alphaE+2.1*(alphaE-pAlphaE)

    -- Pitch channel:
    local betaE=(vx[3]-m[12]) + cmdPitch
    local betaCorr=-0.25*betaE-2.1*(betaE-pBetaE)

    pAlphaE=alphaE
    pBetaE=betaE
    
    -------------------------------------------------
    -- Rotational control
    -- cmdYaw is treated as yaw control input
    -- add yaw-rate damping to suppress self-spin
    -------------------------------------------------
    local linVel, angVel=sim.getVelocity(heli)
    local yawRate=angVel[3]
    local rotCorr=cmdYaw+0.2*yawRate
    
    -------------------------------------------------
    -- Motor mixing
    -------------------------------------------------
    handlePropeller(1,thrust*(1-alphaCorr+betaCorr+rotCorr))
    handlePropeller(2,thrust*(1-alphaCorr-betaCorr-rotCorr))
    handlePropeller(3,thrust*(1+alphaCorr-betaCorr+rotCorr))
    handlePropeller(4,thrust*(1+alphaCorr+betaCorr-rotCorr))
end 


function handlePropeller(index,particleVelocity)
    local propellerRespondable=propellerHandles[index]
    local propellerJoint=jointHandles[index]
    local propeller=sim.getObjectParent(propellerRespondable)
    local particleObject=particleObjects[index]

    local maxParticleDeviation=math.tan(particleScatteringAngle*0.5*math.pi/180)*particleVelocity
    local notFullParticles=0

    local t=sim.getSimulationTime()
    sim.setJointPosition(propellerJoint,t*10)

    local ts=sim.getSimulationTimeStep()
    local m=sim.getObjectMatrix(propeller)

    local particleCnt=0
    local pos={0,0,0}
    local dir={0,0,1}
    
    local requiredParticleCnt=particleCountPerSecond*ts+notFullParticles
    notFullParticles=requiredParticleCnt % 1
    requiredParticleCnt=math.floor(requiredParticleCnt)

    while (particleCnt<requiredParticleCnt) do
        local x=(math.random()-0.5)*2
        local y=(math.random()-0.5)*2
        if (x*x+y*y<=1) then
            if (simulateParticles) then
                pos[1]=x*0.08
                pos[2]=y*0.08
                pos[3]=-particleSize*0.6

                dir[1]=pos[1]+(math.random()-0.5)*maxParticleDeviation*2
                dir[2]=pos[2]+(math.random()-0.5)*maxParticleDeviation*2
                dir[3]=pos[3]-particleVelocity*(1+0.2*(math.random()-0.5))

                pos=sim.multiplyVector(m,pos)
                dir=sim.multiplyVector(m,dir)

                local itemData={pos[1],pos[2],pos[3],dir[1],dir[2],dir[3]}
                sim.addParticleObjectItem(particleObject,itemData)
            end
            particleCnt=particleCnt+1
        end
    end

    -- Apply reactive force onto the body:
    local totalExertedForce=
        particleCnt*particleDensity*particleVelocity*math.pi*particleSize*particleSize*particleSize/(6*ts)

    local force={0,0,totalExertedForce}
    m[4]=0
    m[8]=0
    m[12]=0
    force=sim.multiplyVector(m,force)

    local rotDir=1-math.mod(index,2)*2
    local torque={0,0,rotDir*0.002*particleVelocity}
    torque=sim.multiplyVector(m,torque)

    sim.addForceAndTorque(propellerRespondable,force,torque)
end