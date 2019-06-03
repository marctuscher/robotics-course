

def grasp2dict(res):
    pred = {
        "x": res.grasp.center.x,
        "y": res.grasp.center.y,
        "angle": res.grasp.angle,
        "q": res.q_value,
        "approachAxis": [int(res.grasp.approach_axis[0]), int(res.grasp.approach_axis[1]), int(res.grasp.approach_axis[2])],
        "axis": [res.grasp.axis[0], res.grasp.axis[0]],
        "width": res.grasp.width,
        "depth": res.grasp.depth,
        "approachAngle": res.grasp.approach_angle
    }
    return pred