"""Mixed-agent street scene "last-metre delivery" (domain 1b), writing the INTERFACE.md format.

Top-down, 40 m x 25 m. A two-lane road runs along x through the middle with a zebra crossing at
x = 0, bike lanes on both edges, sidewalks beyond them, a Café on the north side and a Station
entrance on the south side. Agents:

    car         lanes at 8-14 m/s, follows whatever is ahead in its lane (IDM: the car ahead, a
                cyclist drifting out, a person on the road), stops for anyone on the zebra and for
                people waiting at the curb when it can stop comfortably; enters and leaves the map
    pedestrian  the crowd's social-force model (domains/crowd.py) on sidewalks and the zebra; waits
                at the curb until cars are > 3 s away or stopped; some look at their phone (they
                slow down for about 1 s first, then stand 2-5 s)
    cyclist     bike lanes at 4-6 m/s, sometimes drifting out into the car lane for 2-3.5 s
    robot       starts inside the Café at the counter (pickup), leaves through the Café's narrow
                1.2 m door, waiting beside it while a customer comes in; then to the Station door,
                waiting at the zebra while a car or cyclist is close or someone is in its way, then
                shuttles back

Scripted events at fixed times (CrowdConfig-style `events`), logged in the `events` column of the
agents involved and in meta.json:

     0 s  a customer walks into the Café just as the robot reaches the door from inside; the robot
          logs robot_waits_door when it stops for them and robot_door_clear when it goes

    12 s  sudden_stop           a walking pedestrian near the robot stops dead for 3 s
    22 s  car_brake_crosswalk   a pedestrian steps onto the zebra; an approaching car brakes hard
    32 s  jaywalk               a pedestrian crosses mid-block; a car approaching in that lane brakes
    40 s  rush_hour             pedestrians switch to hurried settings, commuters pour out of the
                                Station, twice as many cars (meta: regime_switch_t = 40)

Output is the crowd format plus `agent_type` (car, pedestrian, cyclist, robot), `length`, `width`,
`heading`, `events`, `activity` (what a person or the robot is doing: walk, wait, cross, jaywalk,
phone_slow, phone, stop; "drive" for vehicles; a label for analysis and the 3D viewer, never a model
input), and one-hot `ext_type_car`, `ext_type_cyclist`, `ext_type_pedestrian`: the
core turns numeric ext_* columns into model features, so agent type is a feature with no other
change to the prediction pipeline. `kind` keeps the core's names (person, car, cyclist, robot).
Buildings, road, zebra and doors go into meta.json["map"]; obstacles.csv holds only lamp posts and
planters (the core treats every obstacle as a circle, which suits those but not buildings).

    python -m foresight.domains.street --episodes 5                   # summary per agent type
    python -m foresight.domains.street --episodes 50 --out data/street
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from foresight.domains.crowd import MAX_FORCE, OBST_A, OBST_B, PERSON_R, ROBOT_R, interaction_force, save

# ---------- map (metres; x along the road, y across it) ----------
HALF_W, HALF_H = 20.0, 12.5
LANE_Y = 1.75            # eastbound lane (+x) at y = -1.75, westbound (-x) at y = +1.75
ROAD = 3.5               # car lanes |y| <= 3.5
BIKE_Y = 4.1             # bike lanes: eastbound at y = -4.1, westbound at y = +4.1
CURB = 4.7               # sidewalk for |y| >= CURB
WALL = 8.0               # building fronts at |y| = 8
CROSS_X = 2.0            # zebra |x| <= 2
STOP_GAP = 1.0           # stop line this far before the zebra
CURB_WAIT = 5.2          # |y| where people wait to cross
CAFE_DOOR = np.array([-13.5, 7.5])     # just outside the Café's door
CAFE_DOOR_W = 1.2                       # a tight door: one at a time
DOOR_LINE = np.array([CAFE_DOOR[0], WALL])
ROBOT_START = np.array([-12.8, 10.9])   # inside, at the end of the counter
ROBOT_DOOR_WAIT = np.array([-12.6, 9.3])   # inside, beside the door, out of the customer's way
CUSTOMER_START = np.array([-12.0, 6.5])
COUNTER = np.array([-16.0, 10.7])
DOOR_IN_R, DOOR_COMING_R = 1.0, 2.5     # the robot waits while someone is in the doorway, or within
                                        # 2.5 m and walking towards it
STATION_DOOR = np.array([12.5, -7.5])
EXIT = 24.0              # cars and cyclists enter/leave at |x| = 24 (rows only exported for |x| <= 20)
MAP = dict(
    road=dict(y=[-ROAD, ROAD], lanes=[dict(y=-LANE_Y, direction=1), dict(y=LANE_Y, direction=-1)]),
    bike_lanes=[dict(y=-BIKE_Y, direction=1), dict(y=BIKE_Y, direction=-1)],
    curb_abs_y=CURB, wall_abs_y=WALL,
    crosswalk=dict(x=[-CROSS_X, CROSS_X], y=[-CURB, CURB]),
    buildings=[dict(name="Café", x=[-19.0, -8.0], y=[WALL, HALF_H], door=CAFE_DOOR.tolist(), door_width=CAFE_DOOR_W),
               dict(name="Station", x=[6.0, 19.0], y=[-HALF_H, -WALL], door=STATION_DOOR.tolist())],
)
OBSTACLES = [(-16.0, 5.0, 0.15), (-9.0, 5.0, 0.15), (9.0, 5.0, 0.15), (16.0, 5.0, 0.15),
             (-16.0, -5.0, 0.15), (-9.0, -5.0, 0.15), (9.0, -5.0, 0.15), (16.0, -5.0, 0.15),
             (6.0, 6.9, 0.6), (-6.0, -6.9, 0.6)]   # lamp posts at the curb, two planters

# ---------- behaviour ----------
PED = {   # desired speed, reaction time, social-force strength/range, phone-stop rate (/s)
    "normal":    dict(v0=1.2, tau=0.5,  A=2.0, B=0.5, phone=0.03),
    "rush_hour": dict(v0=1.6, tau=0.35, A=2.0, B=0.3, phone=0.008),
}
CAR_L, CAR_W = 4.5, 1.8
IDM = dict(a=1.5, b=2.5, T=1.2, s0=2.0, max_brake=8.0)
CAR_RATE = {"normal": 0.12, "rush_hour": 0.24}     # arrivals per second per lane
CYCLIST_R = 0.5
SAFE_TTC_CAR, SAFE_TTC_BIKE = 3.0, 2.0              # s, what pedestrians and the robot wait for
CROSS_SHARE = 0.35       # share of new destinations across the road
COURTESY_S = 4.0         # a vehicle held this long at the line no longer yields to people still at the curb
TYPES = ("pedestrian", "car", "cyclist")
COLUMNS = ["episode", "t", "agent_id", "agent_type", "kind", "is_ego", "x", "y", "vx", "vy", "radius",
           "length", "width", "heading", "regime", "goal_x", "goal_y", "collided", "events", "activity",
           "ext_type_pedestrian", "ext_type_car", "ext_type_cyclist"]
KIND = dict(pedestrian="person", car="car", cyclist="cyclist", robot="robot")


@dataclass(frozen=True)
class StreetConfig:
    n_pedestrians: int = 14
    n_cyclists: int = 2              # 0 to leave them out
    duration: float = 60.0
    dt: float = 0.2
    robot_speed: float = 1.0
    phone_share: float = 0.35        # share of pedestrians who sometimes stop to look at their phone
    jitter: float = 0.1
    events: tuple = ((12.0, "sudden_stop"), (22.0, "car_brake_crosswalk"), (32.0, "jaywalk"),
                     (40.0, "rush_hour"))

    @property
    def rush_t(self):
        return next((t for t, e in self.events if e == "rush_hour"), None)


@dataclass
class Agent:
    id: int
    type: str
    p: np.ndarray
    v: np.ndarray
    radius: float
    length: float = 0.0
    width: float = 0.0
    heading: float = 0.0
    # people and the robot
    route: list = field(default_factory=list)      # [(point, kind)], kind: go | curb | other | dest
    mode: str = "walk"                              # walk wait cross jaywalk phone_slow phone stop order wait_door
    mode_left: float = 0.0
    phone_prone: bool = False
    jit: np.ndarray = None
    # cars and cyclists
    direction: int = 1
    lane_y: float = 0.0
    v0: float = 0.0
    drift: float = 0.0
    drift_left: float = 0.0
    held: float = 0.0                               # seconds stopped at the zebra's stop line
    indoor: bool = False                            # the Café customer: no building-front push
    waited: bool = False                            # robot: has waited at the Café door


class StreetSim:
    """One episode. Agent 0 is the robot; other ids are unique for the episode (cars and cyclists
    that leave are replaced by new ids)."""

    def __init__(self, seed: int, config: StreetConfig = StreetConfig()):
        self.seed, self.cfg = seed, config

    # ---------- setup ----------
    def reset(self) -> list[dict]:
        c = self.cfg
        self.rng = rng = np.random.default_rng(self.seed)
        self.k, self.next_id = 0, 1
        self.agents: dict[int, Agent] = {}
        self.pending: dict[int, list[str]] = {}
        self.event_log: list[dict] = []
        self.obs_xy = np.array([o[:2] for o in OBSTACLES]); self.obs_r = np.array([o[2] for o in OBSTACLES])
        robot = Agent(0, "robot", ROBOT_START.copy(), np.zeros(2), ROBOT_R)
        outside = CAFE_DOOR + [0, -0.3]
        robot.route = ([(ROBOT_DOOR_WAIT, "door"), (DOOR_LINE + [0, 0.5], "go"), (outside, "go")]
                       + self._robot_route(outside, STATION_DOOR + [0, 0.3]))
        self.agents[0] = robot
        # a customer walks into the Café just as the robot reaches the door from inside
        cust = self._spawn_pedestrian(CUSTOMER_START.copy())
        cust.route = [(CAFE_DOOR.copy(), "go"), (DOOR_LINE + [0, 1.0], "go"), (COUNTER, "counter")]
        cust.indoor, cust.phone_prone = True, False
        for _ in range(c.n_pedestrians - 1):
            self._spawn_pedestrian(self._sidewalk_point())
        for d in (1, -1):                                   # two cars per lane, spread out
            for x in rng.choice(np.arange(-EXIT + 4, EXIT - 4, 16.0), 2, replace=False):
                self._spawn_car(d, x=float(x) + rng.uniform(-2, 2))
        for i in range(c.n_cyclists):
            self._spawn_cyclist(1 if i % 2 == 0 else -1, x=rng.uniform(-HALF_W, HALF_W))
        self.collided = {}
        return self._state()

    def _new_id(self):
        self.next_id += 1
        return self.next_id - 1

    def _ped_params(self):
        return PED[self._regime()]

    def _regime(self):
        r = self.cfg.rush_t
        return "rush_hour" if r is not None and self.k * self.cfg.dt >= r - 1e-9 else "normal"

    def _sidewalk_point(self, side=None):
        side = side if side is not None else self.rng.choice([-1, 1])
        for _ in range(200):
            p = np.array([self.rng.uniform(-HALF_W + 1, HALF_W - 1), side * self.rng.uniform(CURB + 0.6, WALL - 0.5)])
            if (np.all(np.hypot(*(self.obs_xy - p).T) > self.obs_r + 0.6)
                    and all(np.hypot(*(a.p - p)) > 0.8 for a in self.agents.values() if a.type != "car")):
                return p
        return p

    def _destination(self, here=None):
        """A door, a map end or a sidewalk point; across the road with probability CROSS_SHARE."""
        side = self.rng.choice([-1, 1]) if here is None else (-1 if self.rng.random() < CROSS_SHARE else 1) * np.sign(here[1])
        r = self.rng.random()
        if r < 0.25:
            return (CAFE_DOOR if side > 0 else STATION_DOOR).copy()
        if r < 0.5:
            return np.array([self.rng.choice([-1, 1]) * (HALF_W - 0.8), side * 6.3])
        return self._sidewalk_point(side)

    def _route(self, p, dest, cross_x=None, wait_y=CURB_WAIT):
        """Waypoints from p to dest, via the zebra when dest is across the road."""
        if np.sign(p[1]) == np.sign(dest[1]):
            return [(dest, "dest")]
        xc = cross_x if cross_x is not None else self.rng.uniform(-CROSS_X + 0.7, CROSS_X - 0.7)
        s = np.sign(p[1])
        return [(np.array([xc, s * wait_y]), "curb"), (np.array([xc, -s * wait_y]), "other"), (dest, "dest")]

    def _robot_route(self, p, dest):
        # the robot waits and crosses along the zebra's east edge, a step back from the curb, so it
        # never stands where people step off the crossing
        return self._route(p, dest, cross_x=CROSS_X - 0.4, wait_y=CURB_WAIT + 0.4)

    def _spawn_pedestrian(self, p, dest=None):
        a = Agent(self._new_id(), "pedestrian", np.array(p, float), np.zeros(2), PERSON_R)
        a.jit = self.rng.normal(1.0, self.cfg.jitter, 4).clip(0.7, 1.3)
        a.phone_prone = bool(self.rng.random() < self.cfg.phone_share)
        a.route = self._route(a.p, dest if dest is not None else self._destination(a.p))
        self.agents[a.id] = a
        return a

    def _spawn_car(self, d, x=None, v=None):
        v0 = self.rng.uniform(8, 14)
        x = -d * EXIT if x is None else x
        a = Agent(self._new_id(), "car", np.array([x, -d * LANE_Y]), np.array([d * (v or v0), 0.0]),
                  0.5 * np.hypot(CAR_L, CAR_W), CAR_L, CAR_W, heading=0.0 if d > 0 else np.pi)
        a.direction, a.lane_y, a.v0 = d, -d * LANE_Y, v0
        self.agents[a.id] = a
        return a

    def _spawn_cyclist(self, d, x=None):
        v0 = self.rng.uniform(4, 6)
        x = -d * EXIT if x is None else x
        a = Agent(self._new_id(), "cyclist", np.array([x, -d * BIKE_Y]), np.array([d * v0, 0.0]),
                  CYCLIST_R, 1.8, 0.6, heading=0.0 if d > 0 else np.pi)
        a.direction, a.lane_y, a.v0 = d, -d * BIKE_Y, v0
        self.agents[a.id] = a
        return a

    def _log(self, name, agent_id, note=""):
        self.event_log.append(dict(t=round(self.k * self.cfg.dt, 6), name=name, agent_id=int(agent_id), note=note))
        self.pending.setdefault(agent_id, []).append(name)

    # ---------- scripted events ----------
    def _run_events(self):
        t = self.k * self.cfg.dt
        for te, name in self.cfg.events:
            if abs(t - te) < self.cfg.dt / 2:
                getattr(self, f"_event_{name}")()

    def _walkers(self, on_sidewalk=True):
        return [a for a in self.agents.values() if a.type == "pedestrian" and a.mode == "walk"
                and (not on_sidewalk or abs(a.p[1]) > CURB + 0.3)]

    def _event_sudden_stop(self):
        cands = [a for a in self._walkers() if np.hypot(*a.v) > 0.6] or self._walkers()
        if not cands:
            return
        a = min(cands, key=lambda a: np.hypot(*(a.p - self.agents[0].p)))
        a.mode, a.mode_left = "stop", 3.0
        self._log("sudden_stop", a.id)

    def _event_car_brake_crosswalk(self):
        # a pedestrian steps off the north curb onto the zebra without waiting; the westbound
        # lane (cars moving -x, coming from x > 0) is the first one they cross
        p = self._spawn_pedestrian(np.array([-0.6, CURB + 0.4]), dest=STATION_DOOR + [self.rng.uniform(-2, 2), 0])
        p.route = p.route[1:]                                # skip the curb wait
        p.mode = "cross"
        stop_line = CROSS_X + STOP_GAP
        # an approaching car with room to stop (hard braking, not a crash): an existing one 15-30 m
        # out at speed, else a new one entering at 13 m/s if the lane entry is free
        cars = [a for a in self.agents.values() if a.type == "car" and a.direction == -1
                and 15 < a.p[0] - CAR_L / 2 - stop_line < 30 and abs(a.v[0]) > 7]
        if cars:
            car = min(cars, key=lambda a: a.p[0])
        elif not any(a.type == "car" and a.direction == -1 and 14 < a.p[0] < 30 for a in self.agents.values()):
            car = self._spawn_car(-1, x=21.5, v=13.0)
        else:
            car = min((a for a in self.agents.values() if a.type == "car" and a.direction == -1 and a.p[0] > stop_line),
                      key=lambda a: a.p[0])
        self._log("steps_onto_crosswalk", p.id)
        self._log("car_brake_crosswalk", car.id, note=f"pedestrian {p.id}")

    def _event_jaywalk(self):
        cands = [a for a in self._walkers() if 6 < abs(a.p[0]) < 17]
        if not cands:
            return
        a = cands[int(self.rng.integers(len(cands)))]
        s = np.sign(a.p[1])
        a.route = [(np.array([a.p[0], -s * CURB_WAIT]), "other"), (self._sidewalk_point(-s), "dest")]
        a.mode = "jaywalk"
        d = -1 if s > 0 else 1                               # the lane they step into first
        upstream = a.p[0] - d * 45.0
        if not any(b.type == "car" and b.direction == d and abs(b.p[0] - upstream) < 10 for b in self.agents.values()):
            self._spawn_car(d, x=float(np.clip(upstream, -EXIT - 25, EXIT + 25)), v=12.0)
        self._log("jaywalk", a.id)

    def _event_rush_hour(self):
        for a in self.agents.values():
            if a.type == "pedestrian":
                self.pending.setdefault(a.id, []).append("rush_hour")
        self.event_log.append(dict(t=round(self.k * self.cfg.dt, 6), name="rush_hour", agent_id=-1,
                                   note="pedestrians hurried, commuters from the Station, double car rate"))
        self.commuters_left = 6

    # ---------- dynamics ----------
    def step(self, robot_cmd: tuple[float, float] | None = None) -> tuple[list[dict], dict]:
        """Advance one dt. robot_cmd = (vx, vy) for the robot, or None for the scripted route."""
        c, dt, rng = self.cfg, self.cfg.dt, self.rng
        self._run_events()
        regime = self._regime()
        # arrivals: cars, cyclists, rush-hour commuters
        for d in (1, -1):
            if rng.random() < CAR_RATE[regime] * dt:
                last = [a.p[0] * d for a in self.agents.values() if a.type == "car" and a.direction == d]
                if not last or min(last) > -EXIT + 14:
                    self._spawn_car(d)
        if getattr(self, "commuters_left", 0) and self.k % round(1.0 / dt) == 0:
            self._spawn_pedestrian(STATION_DOOR + [rng.uniform(-1.5, 1.5), 0.3],
                                   dest=CAFE_DOOR if rng.random() < 0.5 else self._destination())
            self.commuters_left -= 1

        waiting = [a for a in self.agents.values() if a.mode == "wait" and a.type in ("pedestrian", "robot")]
        on_zebra = [a for a in self.agents.values() if a.type in ("pedestrian", "robot")
                    and abs(a.p[0]) <= CROSS_X + 0.3 and abs(a.p[1]) < CURB]
        self._move_robot(robot_cmd)
        self._move_people()
        for a in list(self.agents.values()):
            if a.type in ("car", "cyclist"):
                self._move_vehicle(a, waiting, on_zebra)
        for a in list(self.agents.values()):                 # leave at the far end, replace cyclists
            if a.type in ("car", "cyclist") and a.p[0] * a.direction > EXIT:
                del self.agents[a.id]
                if a.type == "cyclist":
                    self._spawn_cyclist(a.direction)
        self.k += 1
        self.collided = self._collisions()
        return self._state(), {"collided": bool(self.collided.get(0, False))}

    def _safe_to_cross(self, a):
        """No car within SAFE_TTC_CAR s (or on the zebra) unless stopped; same for cyclists."""
        for b in self.agents.values():
            if b.type not in ("car", "cyclist"):
                continue
            d = b.direction
            front = b.p[0] * d + b.length / 2
            dist = -CROSS_X - front                          # along its direction, to the zebra's near edge
            if dist < -(2 * CROSS_X + b.length):
                continue                                     # already past
            speed = abs(b.v[0])
            if dist < 0 and abs(b.p[1]) < CURB:
                return False                                 # on the zebra
            if b.held > COURTESY_S:
                return False                                 # it has waited long enough: let it go
            if speed > 0.5 and dist / speed < (SAFE_TTC_CAR if b.type == "car" else SAFE_TTC_BIKE):
                return False
        return True

    def _move_robot(self, cmd):
        r = self.agents[0]
        if cmd is not None:
            r.v = np.asarray(cmd, float)
        else:
            if not r.route:                                  # arrived: shuttle back the other way
                dest = CAFE_DOOR + [0, -0.3] if r.p[1] < 0 else STATION_DOOR + [0, 0.3]
                r.route = self._robot_route(r.p, dest)
            goal, kind = r.route[0]
            to_g = goal - r.p; dg = np.hypot(*to_g)
            others = [a for a in self.agents.values() if a.id != 0 and a.type in ("pedestrian", "cyclist")]
            near = min((np.hypot(*(a.p - r.p)) - a.radius for a in others), default=np.inf)
            speed = self.cfg.robot_speed * np.clip((near - 0.4) / 0.8, 0.0, 1.0)
            if r.mode == "wait_door":                        # inside the Café: is the door free?
                def using_door(a):
                    to_door = DOOR_LINE - a.p; d = np.hypot(*to_door)
                    return d < DOOR_IN_R or (d < DOOR_COMING_R and a.v @ to_door / max(d, 1e-9) > 0.3)
                busy = [a for a in others if a.type == "pedestrian" and using_door(a)]
                if busy:
                    if not r.waited:
                        self._log("robot_waits_door", 0, note=f"pedestrian {busy[0].id}")
                        r.waited = True
                    speed = 0.0
                else:
                    if r.waited:
                        self._log("robot_door_clear", 0)
                    r.mode = "walk"
            if r.mode == "wait":
                ahead = [a for a in others if np.hypot(*(a.p - r.p)) < 1.2 and (a.p - r.p) @ (r.route[0][0] - r.p) > 0]
                if self._safe_to_cross(r) and not ahead:
                    r.mode = "cross"
                speed = 0.0 if r.mode == "wait" else speed
            if dg < 0.3 and r.mode != "wait_door":
                r.route.pop(0)
                r.mode = "wait" if kind == "curb" else "wait_door" if kind == "door" else "walk"
                r.v = np.zeros(2)
            else:
                r.v = to_g / dg * min(speed, dg / self.cfg.dt)
        r.p = np.clip(r.p + r.v * self.cfg.dt, [-HALF_W + 0.3, -HALF_H + 0.3], [HALF_W - 0.3, HALF_H - 0.3])

    def _move_people(self):
        c, dt, rng = self.cfg, self.cfg.dt, self.rng
        pp = self._ped_params()
        group = [a for a in self.agents.values() if a.type in ("pedestrian", "robot", "cyclist")]
        P = np.array([a.p for a in group]); V = np.array([a.v for a in group])
        R = np.array([a.radius for a in group])
        ped = np.array([a.type == "pedestrian" for a in group])
        jit = np.array([a.jit if a.jit is not None else np.ones(4) for a in group])
        v0, A, B = pp["v0"] * jit[:, 0], pp["A"] * jit[:, 1], pp["B"] * jit[:, 2]
        tau = np.full(len(group), pp["tau"])
        E = np.zeros_like(P); v_des = np.zeros(len(group))
        for i, a in enumerate(group):
            if a.type != "pedestrian":
                sp = np.hypot(*a.v); E[i] = a.v / sp if sp > 1e-6 else 0.0
                continue
            if a.mode == "order":                            # at the Café counter, ordering
                v_des[i] = 0.0
                continue
            self._update_mode(a, pp)
            if not a.route:
                a.route = self._route(a.p, self._destination(a.p))
            goal, kind = a.route[0]
            to_g = goal - a.p; dg = np.hypot(*to_g)
            E[i] = to_g / max(dg, 1e-9)
            v_des[i] = {"walk": v0[i], "cross": v0[i] * 1.1, "jaywalk": max(v0[i], 1.3) * 1.15,
                        "wait": 0.0, "phone_slow": v0[i] * max(a.mode_left, 0.0) / 1.0,
                        "phone": 0.0, "stop": 0.0}[a.mode]
            if a.mode == "stop":
                tau[i] = 0.15                                # stops dead, no slowing first
            # at the curb, stop within ~1 m of the spot (queue behind whoever is already waiting);
            # after crossing, arrived as soon as they are back on the sidewalk (never stuck on the
            # zebra because someone, e.g. the waiting robot, stands on their exact spot)
            back_on_sidewalk = kind == "other" and abs(a.p[1]) > CURB and np.sign(a.p[1]) == np.sign(goal[1])
            if dg < (0.9 if kind == "curb" else 0.5) or back_on_sidewalk:
                a.route.pop(0)
                if kind == "curb":
                    a.mode = "wait"
                elif kind == "other":
                    a.mode = "walk"
                elif kind == "counter":
                    a.mode = "order"
        F = (E * v_des[:, None] - V) / tau[:, None]
        F += interaction_force(P, V, E, R, A, B, strong=~ped)  # robot and cyclists: keep extra distance
        # lamp posts and planters
        od = P[:, None, :] - self.obs_xy[None]
        odist = np.maximum(np.linalg.norm(od, axis=2), 1e-6)
        F += (OBST_A * np.exp(-(odist - self.obs_r[None] - R[:, None]) / OBST_B)[..., None] * od / odist[..., None]).sum(axis=1)
        # cars are solid boxes: people on the road step around them (a jaywalker meeting a queue)
        for car in (a for a in self.agents.values() if a.type == "car"):
            half = np.array([car.length / 2, car.width / 2])
            q = np.clip(P, car.p - half, car.p + half)       # nearest point of the car to each person
            dvec = P - q; dist = np.hypot(dvec[:, 0], dvec[:, 1])
            inside = dist < 1e-6
            dvec[inside] = np.c_[np.zeros(inside.sum()), np.sign(P[inside, 1] - car.p[1]) + 1e-9]
            dist[inside] = 0.0
            near = dist < 3.0
            F[near] += (8.0 * np.exp(-(dist[near] - R[near]) / 0.25))[:, None] * dvec[near] / np.maximum(np.hypot(*dvec[near].T), 1e-9)[:, None]
        # building fronts and map ends
        sy = np.sign(P[:, 1]) + (P[:, 1] == 0)
        outdoor = np.array([not a.indoor for a in group])
        F[outdoor, 1] -= sy[outdoor] * 3.0 * np.exp(-(WALL - np.abs(P[outdoor, 1]) - R[outdoor]) / 0.25)
        F[:, 0] += 3.0 * (np.exp(-(P[:, 0] + HALF_W - R) / 0.25) - np.exp(-(HALF_W - P[:, 0] - R) / 0.25))
        for i, a in enumerate(group):
            if not ped[i]:
                continue
            if a.mode not in ("cross", "jaywalk") and abs(a.p[1]) > CURB - 0.3:   # stay on the sidewalk
                F[i, 1] += sy[i] * 6.0 * np.exp(-(abs(a.p[1]) - CURB - R[i]) / 0.2)
            fm = np.hypot(*F[i])
            if fm > MAX_FORCE:
                F[i] *= MAX_FORCE / fm
            a.v = a.v + F[i] * dt
            sp, cap = np.hypot(*a.v), 1.3 * max(v0[i], 0.5) * (1.2 if a.mode == "jaywalk" else 1.0)
            if sp > cap:
                a.v *= cap / sp
            a.p = np.clip(a.p + a.v * dt, [-HALF_W + 0.05, -HALF_H + 0.05], [HALF_W - 0.05, HALF_H - 0.05])

    def _update_mode(self, a, pp):
        dt, rng = self.cfg.dt, self.rng
        if a.mode == "wait" and self._safe_to_cross(a):
            a.mode = "cross"
        elif a.mode == "stop":
            a.mode_left -= dt
            if a.mode_left <= 0:
                a.mode = "walk"
        elif a.mode == "phone_slow":                         # visible slowing, then standing still
            a.mode_left -= dt
            if a.mode_left <= 0:
                a.mode, a.mode_left = "phone", rng.uniform(2.0, 5.0)
        elif a.mode == "phone":
            a.mode_left -= dt
            if a.mode_left <= 0:
                a.mode = "walk"
        elif (a.mode == "walk" and a.phone_prone and abs(a.p[1]) > CURB + 0.3
              and rng.random() < pp["phone"] * a.jit[3] * dt):
            a.mode, a.mode_left = "phone_slow", 1.0

    def _move_vehicle(self, a, waiting, on_zebra):
        """IDM along the lane. Leaders: the vehicle ahead, anyone on the road whose position in 1 s
        is in this lane, and the stop line while someone is on the zebra (or waiting at the curb and
        the vehicle can stop comfortably). Cyclists also drift sideways now and then."""
        dt, d = self.cfg.dt, a.direction
        v = abs(a.v[0])
        leaders = []                                         # (gap to its rear, its speed along d)
        for b in self.agents.values():
            if b.id == a.id:
                continue
            ahead = (b.p[0] - a.p[0]) * d
            if ahead <= 0 or ahead > 60:
                continue
            if b.type in ("car", "cyclist"):
                if abs(b.p[1] - a.p[1]) < (a.width + b.width) / 2 + 0.2:
                    leaders.append((ahead - (a.length + b.length) / 2, max(b.v[0] * d, 0.0)))
            elif abs(b.p[1]) < CURB + 0.2:                   # a person or the robot on the road
                # where they are, or will be in 1 s; a vehicle already on the zebra only stops for
                # someone directly in front of it, so it clears the crossing instead of blocking it
                on_zebra_now = abs(a.p[0]) < CROSS_X + a.length / 2
                y1 = b.p[1] if on_zebra_now else b.p[1] + b.v[1] * 1.0
                if min(abs(b.p[1] - a.p[1]), abs(y1 - a.p[1])) < a.width / 2 + b.radius + 0.4:
                    leaders.append((ahead - a.length / 2 - b.radius, max(b.v[0] * d, 0.0)))
        stop_gap = -(CROSS_X + STOP_GAP) - (a.p[0] * d + a.length / 2)   # front to stop line
        if stop_gap > -0.5:
            comfortable = v * v / (2 * max(stop_gap, 0.1)) < 3.0
            # people on the zebra in this lane, or still heading into it
            in_path = [b for b in on_zebra
                       if abs(b.p[1] - a.p[1]) < a.width / 2 + b.radius + 0.6
                       or (np.sign(a.p[1] - b.p[1]) * b.v[1] > 0.2 and abs(b.p[1] - a.p[1]) < 4.5)]
            # people waiting at the curb on this lane's side (they will enter this lane first)
            mine = [w for w in waiting if np.sign(w.p[1]) == np.sign(a.lane_y)]
            if in_path or (mine and comfortable and a.held <= COURTESY_S):
                leaders.append((max(stop_gap, 0.01), 0.0))
            if v < 0.5 and stop_gap < 6.0:
                a.held += dt                                 # counts until it has passed the zebra
        else:
            a.held = 0.0
        acc = IDM["a"] * (1 - (v / a.v0) ** 4)
        for gap, vl in leaders:
            s_star = IDM["s0"] + v * IDM["T"] + v * (v - vl) / (2 * np.sqrt(IDM["a"] * IDM["b"]))
            acc = min(acc, IDM["a"] * (1 - (v / a.v0) ** 4 - (max(s_star, 0) / max(gap, 0.1)) ** 2))
        acc = float(np.clip(acc, -IDM["max_brake"], IDM["a"]))
        v_new = max(0.0, v + acc * dt)
        a.p = a.p + np.array([d * (v + v_new) / 2 * dt, 0.0])
        vy = 0.0
        if a.type == "cyclist":
            if a.drift_left > 0:
                a.drift_left -= dt
            elif self.rng.random() < 0.05 * dt and not any(   # only with no car alongside
                    b.type == "car" and b.direction == d and -12 < (b.p[0] - a.p[0]) * d < 6
                    for b in self.agents.values()):
                a.drift, a.drift_left = d * self.rng.uniform(0.8, 1.3), self.rng.uniform(2.0, 3.5)
            target = a.lane_y + (a.drift if a.drift_left > 0 else 0.0)
            vy = float(np.clip((target - a.p[1]) / 0.8, -0.7, 0.7))
            a.p[1] += vy * dt
        a.v = np.array([d * v_new, vy])

    def _collisions(self):
        ids = list(self.agents)
        hit = {i: False for i in ids}
        for n, i in enumerate(ids):
            a = self.agents[i]
            for j in ids[n + 1:]:
                b = self.agents[j]
                if a.type == "car" and b.type == "car":
                    continue
                if "car" in (a.type, b.type):
                    car, o = (a, b) if a.type == "car" else (b, a)
                    touch = (abs(o.p[0] - car.p[0]) < car.length / 2 + o.radius
                             and abs(o.p[1] - car.p[1]) < car.width / 2 + o.radius)
                else:
                    touch = np.hypot(*(a.p - b.p)) < a.radius + b.radius
                if touch:
                    hit[i] = hit[j] = True
        for i in ids:
            a = self.agents[i]
            if a.type != "car" and np.any(np.hypot(*(self.obs_xy - a.p).T) < self.obs_r + a.radius - 1e-6):
                hit[i] = True
        return hit

    def _state(self) -> list[dict]:
        t = round(self.k * self.cfg.dt, 6)
        regime = self._regime()
        rows = []
        for a in self.agents.values():
            if abs(a.p[0]) > HALF_W:
                continue
            sp = np.hypot(*a.v)
            if sp > 0.1:
                a.heading = float(np.arctan2(a.v[1], a.v[0]))
            goal = a.route[0][0] if a.route else (np.array([a.direction * EXIT, a.lane_y]) if a.type in ("car", "cyclist") else a.p)
            rows.append(dict(
                t=t, agent_id=a.id, agent_type=a.type, kind=KIND[a.type], is_ego=int(a.id == 0),
                x=a.p[0], y=a.p[1], vx=a.v[0], vy=a.v[1], radius=a.radius, length=a.length, width=a.width,
                heading=a.heading, regime="robot" if a.id == 0 else regime, goal_x=goal[0], goal_y=goal[1],
                collided=int(self.collided.get(a.id, False)), events=";".join(self.pending.pop(a.id, [])),
                activity=a.mode if a.type in ("pedestrian", "robot") else "drive",
                **{f"ext_type_{ty}": int(a.type == ty) for ty in TYPES}))
        return rows


# ---------- episodes and datasets ----------
def simulate(seed: int, config: StreetConfig = StreetConfig(), episode: int = 0):
    """Run one episode. Returns (tracks, obstacles, episode_meta); ground truth, no noise."""
    sim = StreetSim(seed, config)
    rows = sim.reset()
    for _ in range(int(round(config.duration / config.dt))):
        rows += sim.step()[0]
    tracks = pd.DataFrame(rows)
    tracks.insert(0, "episode", episode)
    tracks = tracks[COLUMNS].sort_values(["episode", "t", "agent_id"]).reset_index(drop=True)
    obstacles = pd.DataFrame([dict(episode=episode, obstacle_id=i, shape="circle", x=x, y=y, r=r,
                                   w=np.nan, h=np.nan, angle=np.nan) for i, (x, y, r) in enumerate(OBSTACLES)])
    rush = config.rush_t
    meta = dict(seed=seed, duration=config.duration, regime_switch_t=rush,
                regime_before="normal", regime_after="rush_hour" if rush is not None else None,
                events=sim.event_log)
    return tracks, obstacles, meta


def generate(n_episodes: int, seed: int = 0, config: StreetConfig = StreetConfig()):
    """N episodes, same scripted event times, different people, cars and cyclists. Same seed, same data."""
    rng = np.random.default_rng(seed)
    tracks, obstacles, eps = [], [], {}
    for e in range(n_episodes):
        tr, ob, me = simulate(int(rng.integers(2**31)), config, episode=e)
        tracks.append(tr); obstacles.append(ob); eps[str(e)] = me
    meta = dict(sim="street-v1", dt=config.dt, bounds=[-HALF_W, -HALF_H, HALF_W, HALF_H], episodes=eps,
                map=MAP, config={k: v for k, v in asdict(config).items()},
                notes="last-metre delivery: cars (IDM, yield at the zebra), pedestrians (social force), "
                      "cyclists (bike lanes, drift out), robot on a fixed Café -> Station route")
    return pd.concat(tracks, ignore_index=True), pd.concat(obstacles, ignore_index=True), meta


def summary(tracks: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Per agent type: how many, how fast, how often stopped, how wrong constant velocity is."""
    dt = float(meta["dt"])
    tr = tracks.sort_values(["episode", "agent_id", "t"]).copy()
    tr["speed"] = np.hypot(tr.vx, tr.vy)
    g = tr.groupby(["episode", "agent_id"])
    for h in (1, 2):
        k = round(h / dt)
        same = np.isclose(g.t.shift(-k) - tr.t, h)          # the agent is still on the map h s later
        err = np.hypot(g.x.shift(-k) - (tr.x + tr.vx * h), g.y.shift(-k) - (tr.y + tr.vy * h))
        tr[f"cv_err_{h}s"] = err.where(same)
    rows = []
    for ty, d in tr.groupby("agent_type"):
        rows.append(dict(agent_type=ty, agents=d.groupby(["episode", "agent_id"]).ngroups,
                         rows=len(d), mean_speed=d.speed.mean(), p95_speed=d.speed.quantile(0.95),
                         stopped_share=(d.speed < 0.2).mean(), cv_err_1s=d.cv_err_1s.mean(),
                         cv_err_2s=d.cv_err_2s.mean(), collision_rows=int(d.collided.sum())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--cyclists", type=int, default=2)
    ap.add_argument("--out", help="write tracks.csv, obstacles.csv, meta.json here")
    a = ap.parse_args()
    cfg = StreetConfig(duration=a.duration, n_cyclists=a.cyclists)
    tracks, obstacles, meta = generate(a.episodes, a.seed, cfg)
    pd.set_option("display.width", 180)
    print(f"{a.episodes} episodes of {a.duration:g} s, seed {a.seed}\n")
    print(summary(tracks, meta).round(2).to_string(index=False))
    ev = pd.DataFrame([dict(episode=int(e), **x) for e, m in meta["episodes"].items() for x in m["events"]])
    if len(ev):
        print("\nscripted events per episode (agent_id -1 = whole scene):")
        print(ev.pivot_table(index="name", columns="episode", values="t", aggfunc="count", fill_value=0).to_string())
    if a.out:
        save(a.out, tracks, obstacles, meta)
        print(f"\nwrote {a.episodes} episodes to {a.out}")


if __name__ == "__main__":
    main()
