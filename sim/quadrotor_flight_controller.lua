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

    -------------------------------------------------
    -- Betaflight angle mode parameters
    -------------------------------------------------
    -- Original hover particle velocity was 5.45
    -- PWM 1500 (50%) -> 5.45, PWM 2000 (100%) -> 10.9
    max_thrust_vel=10.9

    -- Max commanded tilt (rad) — matches original keyboard range ~0.30
    max_angle=0.30

    -- Max yaw command — matches original keyboard range ~0.30
    max_yaw_cmd=0.30

    -- Linear drag coefficient (simulates air resistance)
    -- At full throttle, excess thrust ≈ weight (2:1 TWR), so excess force ≈ mg ≈ 0.005 N
    -- For terminal velocity ~3 m/s: kd = 0.005/3 ≈ 0.0017
    drag_kd=1.0

    -------------------------------------------------
    -- Object handles
    -------------------------------------------------
    d=sim.getObject('../base')
    heli=sim.getObject('..')

    propellerHandles={}
    jointHandles={}
    particleObjects={-1,-1,-1,-1}

    local ttype=sim.particle_roughspheres
               + sim.particle_cyclic
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

    -- External command signals (PWM 1000-2000)
    sigRoll='cmd_roll'
    sigPitch='cmd_pitch'
    sigYaw='cmd_yaw'
    sigThrust='cmd_thrust'

    -- Default PWM values (neutral/idle)
    cmdRollPWM=1500
    cmdPitchPWM=1500
    cmdYawPWM=1500
    cmdThrustPWM=1000

    -- Horizontal control memory (from original):
    pAlphaE=0
    pBetaE=0

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

    -------------------------------------------------
    -- Read PWM commands from Python
    -------------------------------------------------
    local r=sim.getFloatSignal(sigRoll)
    local p=sim.getFloatSignal(sigPitch)
    local y=sim.getFloatSignal(sigYaw)
    local tCmd=sim.getFloatSignal(sigThrust)

    if r~=nil then cmdRollPWM=r end
    if p~=nil then cmdPitchPWM=p end
    if y~=nil then cmdYawPWM=y end
    if tCmd~=nil then cmdThrustPWM=tCmd end

    -------------------------------------------------
    -- Convert PWM to internal commands
    -------------------------------------------------
    -- Thrust: PWM 1000->0, 1500->5.45 (hover), 2000->10.9
    local thrust_frac=(cmdThrustPWM-1000)/1000
    if thrust_frac<0 then thrust_frac=0 end
    if thrust_frac>1 then thrust_frac=1 end
    local thrust=thrust_frac*max_thrust_vel

    -- Roll/pitch: PWM 1500->0, range +-max_angle
    local cmdRoll=(cmdRollPWM-1500)/500*max_angle
    local cmdPitch=(cmdPitchPWM-1500)/500*max_angle

    -- Yaw: PWM 1500->0, range +-max_yaw_cmd
    local cmdYaw=(cmdYawPWM-1500)/500*max_yaw_cmd

    -------------------------------------------------
    -- Horizontal control (ORIGINAL gains and method)
    -- Uses body-frame matrix for attitude measurement
    -------------------------------------------------
    local m=sim.getObjectMatrix(d)

    local vx={1,0,0}
    vx=sim.multiplyVector(m,vx)

    local vy={0,1,0}
    vy=sim.multiplyVector(m,vy)

    -- Roll channel (original: P=0.25, D=2.1):
    local alphaE=(vy[3]-m[12]) - cmdRoll
    local alphaCorr=0.25*alphaE+2.1*(alphaE-pAlphaE)

    -- Pitch channel (original: P=0.25, D=2.1):
    local betaE=(vx[3]-m[12]) + cmdPitch
    local betaCorr=-0.25*betaE-2.1*(betaE-pBetaE)

    pAlphaE=alphaE
    pBetaE=betaE

    -------------------------------------------------
    -- Yaw control (ORIGINAL: cmdYaw + 0.5*yawRate damping)
    -------------------------------------------------
    local linVel, angVel=sim.getVelocity(heli)
    local yawRate=angVel[3]
    local rotCorr=cmdYaw+0.5*yawRate

    -------------------------------------------------
    -- Air resistance (linear drag) — split across 4 propeller respondables
    -------------------------------------------------
    local dragPerMotor={-drag_kd*linVel[1]/4,-drag_kd*linVel[2]/4,-drag_kd*linVel[3]/4}
    for i=1,4 do
        sim.addForceAndTorque(propellerHandles[i],dragPerMotor,{0,0,0})
    end

    -------------------------------------------------
    -- Motor mixing (ORIGINAL multiplicative)
    -- Cutoff below PWM 1050 to prevent ground interaction
    -------------------------------------------------
    if cmdThrustPWM<1050 then
        pAlphaE=0
        pBetaE=0
        handlePropeller(1,0)
        handlePropeller(2,0)
        handlePropeller(3,0)
        handlePropeller(4,0)
    else
        handlePropeller(1,thrust*(1-alphaCorr+betaCorr+rotCorr))
        handlePropeller(2,thrust*(1-alphaCorr-betaCorr-rotCorr))
        handlePropeller(3,thrust*(1+alphaCorr-betaCorr+rotCorr))
        handlePropeller(4,thrust*(1+alphaCorr+betaCorr-rotCorr))
    end

    -- DEBUG: log every 25 steps (~2Hz)
    if debugCnt==nil then debugCnt=0 end
    debugCnt=debugCnt+1
    if debugCnt%25==0 and cmdThrustPWM>=1050 then
        print(string.format("PWM=%d thr=%.2f pos_z=%.2f vel_z=%.2f drag_z=%.4f",
            cmdThrustPWM, thrust, pos[3], linVel[3], -drag_kd*linVel[3]))
    end
end


function handlePropeller(index,particleVelocity)
    local propellerRespondable=propellerHandles[index]
    local propellerJoint=jointHandles[index]
    local propeller=sim.getObjectParent(propellerRespondable)
    local particleObject=particleObjects[index]

    -- Skip everything when motors are off: no particles, no forces, no joint spin
    if particleVelocity<=0 then
        return
    end

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
