sim = require 'sim'

function sysCall_init()
    motorLeft = sim.getObject("../leftMotor")
    motorRight = sim.getObject("../rightMotor")

    sigV = 'ground_v'
    v = 0.0
end

function sysCall_actuation()
    local cmdV = sim.getFloatSignal(sigV)
    if cmdV ~= nil then
        v = cmdV
    end

    sim.setJointTargetVelocity(motorLeft, v)
    sim.setJointTargetVelocity(motorRight, v)
end

function sysCall_cleanup()
end