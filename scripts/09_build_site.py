"""Generate the walkthrough site with videos inlined as data URIs (the artifact CSP
blocks every external host, so a src="file.mp4" would silently fail)."""
import base64, json, pathlib

V = pathlib.Path("artifacts/videos_web")
def vid(name):
    return base64.b64encode((V / name).read_bytes()).decode()

def player(name, left, right, caption, verdict_l, verdict_r):
    return f'''<figure class="vid">
  <video controls autoplay muted loop playsinline preload="metadata"
         src="data:video/mp4;base64,{vid(name)}"></video>
  <div class="vid-split">
    <span class="side"><b>left</b> {left} <span class="tag {'win' if verdict_l else 'lose'}">{'success' if verdict_l else 'failure'}</span></span>
    <span class="side"><b>right</b> {right} <span class="tag {'win' if verdict_r else 'lose'}">{'success' if verdict_r else 'failure'}</span></span>
  </div>
  <figcaption>{caption}</figcaption>
</figure>'''

sweep = json.load(open("artifacts/ablation_horizon_sweep.json"))
rows = sorted(sweep["rows"], key=lambda r: int(r["condition"].split("=")[1]))
H = [int(r["condition"].split("=")[1]) for r in rows]
S = [r["success_rate"] for r in rows]
N = [(r["n_success"], r["n_episodes"]) for r in rows]

X0, X1, Y0, Y1 = 34, 404, 12, 112
x = lambda h: round(X0 + (h / 100) * (X1 - X0), 1)
y = lambda s: round(Y1 - (s / 0.55) * (Y1 - Y0), 1)
pts = " ".join(f"{x(h)},{y(s)}" for h, s in zip(H, S))
dots = "".join(f'<circle cx="{x(h)}" cy="{y(s)}" r="3"/>' for h, s in zip(H, S))
labs = "".join(f'<text x="{x(h)}" y="124" text-anchor="middle" fill="var(--muted)" '
               f'font-family="ui-monospace,monospace" font-size="7">{h}</text>' for h in H)
curve = f'''<svg viewBox="0 0 416 134" role="img" aria-label="Success against committed horizon: zero below 20 steps, then flat between 30 and 49 percent up to 100.">
  <line x1="{X0}" y1="{Y0}" x2="{X1}" y2="{Y0}" stroke="var(--rule-2)"/>
  <line x1="{X0}" y1="{y(.25)}" x2="{X1}" y2="{y(.25)}" stroke="var(--rule-2)" stroke-dasharray="2 4"/>
  <line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="var(--rule)"/>
  <rect x="{x(41)}" y="{Y0}" width="{x(56)-x(41)}" height="{Y1-Y0}" fill="var(--onset-wash)"/>
  <polyline fill="none" stroke="var(--rev)" stroke-width="2.4" stroke-linejoin="round" points="{pts}"/>
  <g fill="var(--rev)">{dots}</g>{labs}
  <text x="4" y="{Y0+4}" fill="var(--muted)" font-family="ui-monospace,monospace" font-size="7">55%</text>
  <text x="4" y="{Y1+2}" fill="var(--muted)" font-family="ui-monospace,monospace" font-size="7">0%</text>
</svg>'''

sweep_rows = "".join(
    f'<tr{" class=hi" if h == 40 else ""}><td>{h}</td><td>{a}/{b} &nbsp;{s:.0%}</td></tr>'
    for h, s, (a, b) in zip(H, S, N))

html = f'''<title>Reversibility Walkthrough</title>
<style>
:root{{
  --rev:#2a6fdb; --onset:#c8384f; --pass:#2f7d52; --warn:#a8760f;
  --ink:#0e141c; --ink-2:#39424f; --muted:#69737f;
  --paper:#f2f5f9; --card:#fff; --rule:#d8dee7; --rule-2:#e8edf3;
  --rev-wash:rgba(42,111,219,.10); --onset-wash:rgba(200,56,79,.10);
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --col:68ch;
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --rev:#6a9df0; --onset:#e8697d; --pass:#5aa87b; --warn:#d0a03f;
  --ink:#e6ecf4; --ink-2:#b3becd; --muted:#8592a3;
  --paper:#0e141c; --card:#151d28; --rule:#2a3543; --rule-2:#1e2833;
  --rev-wash:rgba(106,157,240,.14); --onset-wash:rgba(232,105,125,.14);
}}}}
:root[data-theme="dark"]{{
  --rev:#6a9df0; --onset:#e8697d; --pass:#5aa87b; --warn:#d0a03f;
  --ink:#e6ecf4; --ink-2:#b3becd; --muted:#8592a3;
  --paper:#0e141c; --card:#151d28; --rule:#2a3543; --rule-2:#1e2833;
  --rev-wash:rgba(106,157,240,.14); --onset-wash:rgba(232,105,125,.14);
}}
*{{box-sizing:border-box}}
body{{background:var(--paper);color:var(--ink);font-family:var(--serif);
  font-size:17px;line-height:1.62;margin:0;padding:0 20px 96px;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:var(--col);margin:0 auto}}
.wide{{max-width:min(96vw,1040px);margin:0 auto}}
h1,h2,h3{{text-wrap:balance;line-height:1.15;margin:0}}
h1{{font-size:clamp(2.4rem,6vw,3.7rem);font-weight:600;letter-spacing:-.022em}}
h2{{font-size:1.55rem;font-weight:600;letter-spacing:-.012em}}
h3{{font-size:1.05rem;font-weight:600}}
p{{margin:0}}
.eyebrow{{font-family:var(--mono);font-size:.72rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--muted)}}
.num{{font-family:var(--mono);font-variant-numeric:tabular-nums}}
section{{padding-top:58px}}
.stack{{display:flex;flex-direction:column;gap:18px}}
.stack-sm{{display:flex;flex-direction:column;gap:10px}}
.lede{{font-size:1.28rem;color:var(--ink-2);max-width:60ch}}
.lede b{{color:var(--ink);font-weight:600}}
/* video */
.vid{{margin:0;background:var(--card);border:1px solid var(--rule);border-radius:3px;overflow:hidden}}
.vid video{{display:block;width:100%;height:auto;background:#000}}
.vid-split{{display:flex;flex-wrap:wrap;gap:6px 20px;padding:11px 15px 0;
  font-family:var(--mono);font-size:.75rem;color:var(--ink-2)}}
.vid-split b{{color:var(--muted);font-weight:400}}
.tag{{font-size:.68rem;letter-spacing:.06em;text-transform:uppercase;
  padding:1px 6px;border-radius:2px;border:1px solid currentColor;margin-left:4px}}
.tag.win{{color:var(--pass)}} .tag.lose{{color:var(--onset)}}
figcaption{{padding:9px 15px 14px;font-size:.92rem;color:var(--ink-2)}}
.vids{{display:flex;flex-direction:column;gap:22px}}
/* tables */
.scroll{{overflow-x:auto;border:1px solid var(--rule);border-radius:3px;background:var(--card)}}
table{{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:.82rem;
  font-variant-numeric:tabular-nums}}
th,td{{padding:9px 14px;text-align:left;white-space:nowrap;border-bottom:1px solid var(--rule-2)}}
thead th{{font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);
  font-weight:400;border-bottom:1px solid var(--rule)}}
tbody tr:last-child td{{border-bottom:0}}
tr.hi td{{background:var(--rev-wash)}}
td.lead{{white-space:normal;min-width:15ch}}
.sig{{color:var(--pass)}} .ns{{color:var(--muted)}}
.note{{border-left:2px solid var(--rule);padding:2px 0 2px 18px;color:var(--ink-2)}}
.note.warn{{border-color:var(--onset)}}
.bignum{{font-family:var(--mono);font-size:2.5rem;letter-spacing:-.03em;line-height:1;color:var(--rev)}}
.bignum.bad{{color:var(--onset)}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:24px}}
.metric{{display:flex;flex-direction:column;gap:6px}}
.metric .lab{{font-family:var(--mono);font-size:.71rem;color:var(--muted);
  letter-spacing:.08em;text-transform:uppercase}}
.card{{background:var(--card);border:1px solid var(--rule);border-radius:3px;padding:20px 22px}}
ul{{margin:0;padding-left:1.1em;color:var(--ink-2)}}
li{{margin:8px 0}} li::marker{{color:var(--muted)}}
code{{font-family:var(--mono);font-size:.86em;background:var(--rule-2);padding:1px 5px;border-radius:2px}}
ol.steps{{margin:0;padding-left:0;list-style:none;counter-reset:s;
  display:flex;flex-direction:column;gap:14px}}
ol.steps li{{counter-increment:s;padding-left:38px;position:relative;color:var(--ink-2);margin:0}}
ol.steps li::before{{content:counter(s);position:absolute;left:0;top:1px;
  font-family:var(--mono);font-size:.72rem;color:var(--rev);
  border:1px solid var(--rule);border-radius:2px;width:24px;height:24px;
  display:grid;place-items:center}}
footer{{margin-top:76px;padding-top:26px;border-top:1px solid var(--rule);
  color:var(--muted);font-family:var(--mono);font-size:.76rem}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}
</style>

<header class="wrap stack">
  <p class="eyebrow">FAACT v2 &middot; ALOHA transfer-cube &middot; MuJoCo</p>
  <h1>Watching a robot pass the point of no return</h1>
  <p class="lede">An action-chunking policy commits to 100 actions at a time and is blind while it
  executes them. We measured <b>reversibility</b> &mdash; the chance the task still succeeds if you
  replan right now &mdash; and used it to decide how long to commit. <b>It didn&rsquo;t work.</b>
  This page walks through what we built, what you can see happening, and why the idea fails.</p>
</header>

<section class="wrap stack">
  <p class="eyebrow">Background</p>
  <h2>What an action-chunking policy is, and why it has a problem</h2>
  <p style="color:var(--ink-2)">A robot policy normally picks one action, looks at the world,
  then picks the next. <b>Action chunking</b> changes that: the policy predicts a whole
  sequence &mdash; here 100 actions, about two seconds of motion &mdash; and executes it
  open-loop before looking again. ACT, the policy used throughout this project, works this
  way and it is why it produces smooth, human-like motion instead of twitchy
  step-by-step corrections.</p>
  <p style="color:var(--ink-2)">The cost is reactivity. <b>While executing a chunk, the robot
  is blind.</b> Knock the cube at step 3 and nothing can respond until step 100. So how long
  should it commit for? Commit long and you get smooth motion that ignores surprises; commit
  short and you can react, but you interrupt the motion constantly.</p>
  <div class="card stack-sm">
    <h3>The idea this project tests</h3>
    <p style="color:var(--ink-2)">Let the robot decide how long to commit based on how much
    trouble it is in. Specifically, on <b>reversibility</b> &mdash; the probability the task
    still succeeds if you stop and replan right now. High reversibility means a mistake is
    still fixable, so commit long. Low reversibility means the point of no return is close, so
    commit short and stay reactive.</p>
    <p class="num" style="color:var(--ink-2);font-size:.95rem">h = clip(h_min, 100, 100 &times;
    R&#770;<sup>&gamma;</sup>)</p>
  </div>
  <p class="note">The appeal is that one quantity would govern three separate problems that
  the literature solves separately: how long to commit, when to intervene, and when to give
  up. This page reports what happened when we measured it.</p>
  <p class="note warn"><b>Why this rebuild exists.</b> A previous attempt measured 0 successes
  out of 20 and zero recoveries. The cause was not the method: its perturbations were never
  actually implemented, so nothing was ever disturbed and there was no headroom above a floor
  of zero. Everything here is built so that cannot happen quietly &mdash; every disturbance is
  verified to change the world, the baseline is verified against the reference implementation
  episode by episode, and each acceptance check is fixed before the run.</p>
</section>

<section class="wide stack">
  <div class="wrap stack-sm">
    <p class="eyebrow">Watch it &middot; 1 of 2</p>
    <h2>The cliff: replanning too often destroys the policy</h2>
    <p style="color:var(--ink-2)">Identical seed, identical disturbance. The only difference is how
    many steps of each predicted chunk the robot commits to before replanning. On the left it
    replans every 5 steps; on the right, every 40.</p>
  </div>
  <div class="vids">
    {player("cliff_seed9203_actuation_noise.mp4", "commits 5 steps at a time", "commits 40 steps at a time",
            "The left arm barely moves at all. An ACT chunk begins at the current pose, so committing only its first 5 actions and replanning from a barely-changed state re-emits a near-identical chunk &mdash; the arm creeps at a fraction of normal speed and runs out of time. Measured net joint displacement over an episode: <b>0.16 at h=5 against 2.02 at h=100</b>. Across 40 episodes h=5 succeeds <b>zero</b> times.",
            False, True)}
    {player("cliff_seed9201_actuation_noise.mp4", "commits 5 steps at a time", "commits 40 steps at a time",
            "A second pair, same comparison. The stall is deterministic: h=5 produced a net displacement of 0.15&ndash;0.17 on every one of five seeds tested, while h=100 moved 1.93&ndash;2.05 on all of them. This is the failure mode behind the 0/40, not bad luck on one seed.",
            False, True)}
  </div>
</section>

<section class="wide stack">
  <div class="wrap stack-sm">
    <p class="eyebrow">Watch it &middot; 2 of 2</p>
    <h2>The method itself: reversibility-gated against a fixed horizon</h2>
    <p style="color:var(--ink-2)">Now the comparison the project was built to make. Left commits a
    full 100 steps every time. Right varies its commitment from 20 to 100 based on predicted
    reversibility &mdash; shortening when it thinks recovery is becoming unlikely.</p>
  </div>
  <div class="vids">
    {player("gated_seed8500_object_displace.mp4", "fixed, 100 steps", "reversibility-gated, 20&ndash;100",
            "The cube is teleported 2&nbsp;cm mid-grasp. The gated controller detects the drop in reversibility and shortens its commitment &mdash; and still fails, while the policy that simply committed and carried on succeeds.",
            True, False)}
    {player("gated_seed8505_actuation_noise.mp4", "fixed, 100 steps", "reversibility-gated, 20&ndash;100",
            "Gaussian noise on the joint targets during transport. Same pattern. Across 79 episodes the gated controller wins 8 and loses 9 against the fixed horizon &mdash; <span class='num'>p&nbsp;=&nbsp;1.000</span>, a dead heat.",
            True, False)}
  </div>
  <div class="wrap">
    <p class="note">These clips are chosen from pairs where the two controllers <em>disagreed</em>,
    because a pair where both succeed shows nothing. That makes them illustrations of the failure
    mode, not evidence about the average case &mdash; the tables below carry that.</p>
  </div>
</section>

<section class="wrap stack">
  <p class="eyebrow">How it works</p>
  <h2>Measuring reversibility instead of guessing it</h2>
  <p style="color:var(--ink-2)">Reversibility is defined operationally, so it can be measured rather
  than assumed. At a given moment in an episode:</p>
  <ol class="steps">
    <li>Snapshot the entire simulator state &mdash; joint positions, velocities, contacts, and the
    constraint solver&rsquo;s warm-start cache.</li>
    <li>Restore that snapshot and let the policy replan from scratch, running to the end of the
    episode. Repeat 8 times with small variation in execution.</li>
    <li>Record the fraction that still succeed. That fraction <em>is</em> R at that state.</li>
    <li>Repeat every 25 steps, across 158 episodes, for <b>1,990 labelled states</b> in total.</li>
  </ol>
  <div class="metrics">
    <div class="metric"><span class="lab">R before disturbance</span><span class="bignum">0.78</span></div>
    <div class="metric"><span class="lab">R after disturbance</span><span class="bignum bad">0.24</span></div>
    <div class="metric"><span class="lab">predicted out-of-fold</span><span class="bignum">&rho; 0.69</span></div>
  </div>
  <p class="note">Reversibility itself came through cleanly. It collapses when the robot is
  disturbed, and a small model predicts it on held-out episodes at &rho;=0.69. <b>The signal was
  never the problem.</b></p>
</section>

<section class="wide stack">
  <div class="wrap stack-sm">
    <p class="eyebrow">Why it fails</p>
    <h2>The horizon is the wrong thing to control</h2>
    <p style="color:var(--ink-2)">Sweeping seven fixed commitment horizons over one set of episodes
    gives a cliff and a plateau. Below about 20 steps the arm stalls and the policy collapses.
    Above it, the horizon does not measurably change anything at all.</p>
  </div>
  <div class="wrap"><div class="card">{curve}
    <p style="font-family:var(--mono);font-size:.74rem;color:var(--muted);margin-top:12px">
    success vs committed horizon (n=37) &middot; shaded band = where every gated controller
    actually operated</p>
  </div></div>
  <div class="wrap stack-sm">
    <div class="scroll"><table>
      <thead><tr><th>Horizon</th><th>Success (n=37)</th></tr></thead>
      <tbody>{sweep_rows}</tbody>
    </table></div>
    <p class="note"><b>Every controller we built operated at a mean horizon of 41&ndash;56</b> &mdash;
    inside the plateau. A controller can only lose by approaching the cliff, and cannot gain by moving
    around within the flat, no matter how well informed it is. That is why an oracle handed
    ground-truth reversibility does no better than a fixed horizon.</p>
    <p class="note warn">Statistically: h=5 vs h=40 gives <span class="num">p&nbsp;=&nbsp;0.00001</span>;
    every comparison above h=20 gives <span class="num">p&nbsp;&ge;&nbsp;0.092</span>. The cliff
    comparisons are the <em>only</em> results in this project that survive correction for multiple
    testing.</p>
  </div>
</section>

<section class="wide stack">
  <div class="wrap stack-sm">
    <p class="eyebrow">The full record</p>
    <h2>Every stage, and what it measured</h2>
    <p style="color:var(--ink-2)">Each stage had a blocking acceptance check fixed in advance.
    Two did not pass, and are reported as failures.</p>
  </div>
  <div class="scroll"><table>
    <thead><tr><th>Stage</th><th>What it does</th><th>Measured</th><th>Gate</th></tr></thead>
    <tbody>
      <tr><td class="lead">Environment</td><td class="lead">headless MuJoCo, video decode, EGL</td><td>6/6 checks</td><td><span class="chip pass">pass</span></td></tr>
      <tr><td class="lead">ACT baseline</td><td class="lead">reproduce the reference eval seed-for-seed</td><td>48/50 agree &middot; 76.0% vs 76.0%</td><td><span class="chip pass">pass</span></td></tr>
      <tr><td class="lead">Disturbances</td><td class="lead">must land in 25&ndash;65% <em>and</em> drop success &ge;15pt</td><td>2 of 4 kinds effective</td><td><span class="chip fail">fail</span></td></tr>
      <tr><td class="lead">Reversibility labels</td><td class="lead">snapshot, replan 8&times;, count successes</td><td>0.78 &rarr; 0.24 &middot; 1,990 states</td><td><span class="chip pass">pass</span></td></tr>
      <tr><td class="lead">Reversibility model</td><td class="lead">predict R from held-out episodes</td><td>&rho; = 0.693 &middot; 5-fold CV</td><td><span class="chip pass">pass</span></td></tr>
      <tr><td class="lead">Horizon control</td><td class="lead">fixed vs learned vs oracle</td><td>no gain over a fixed horizon</td><td><span class="chip fail">fail</span></td></tr>
    </tbody>
  </table></div>
  <div class="wrap stack-sm">
    <h3 style="margin-top:14px">The controller comparison in full</h3>
    <p style="color:var(--ink-2)">All six on the same 76 episodes with identical disturbances.
    The learned rows are scored by a model from the cross-validation fold that never trained on
    that episode, so they are directly comparable to the oracle.</p>
  </div>
  <div class="scroll"><table>
    <thead><tr><th>Controller</th><th>Success</th><th>Replans/ep</th><th>Mean horizon</th><th>vs fixed h=100</th></tr></thead>
    <tbody>
      <tr class="hi"><td class="lead"><b>fixed h=100 &mdash; as published</b></td><td><b>36/76 &nbsp;47%</b></td><td>3.6</td><td>100</td><td class="ns">&mdash;</td></tr>
      <tr><td class="lead">learned gating, &gamma;=1</td><td>35/76 &nbsp;46%</td><td>7.6</td><td>50</td><td class="ns">p = 1.000</td></tr>
      <tr><td class="lead">oracle R, &gamma;=1</td><td>30/76 &nbsp;39%</td><td>8.8</td><td>43</td><td class="ns">p = 0.109</td></tr>
      <tr><td class="lead">oracle R, &gamma;=2</td><td>30/76 &nbsp;39%</td><td>9.2</td><td>41</td><td class="ns">p = 0.109</td></tr>
      <tr><td class="lead">learned gating, &gamma;=2</td><td>30/76 &nbsp;39%</td><td>9.3</td><td>41</td><td class="ns">p = 0.238</td></tr>
      <tr><td class="lead">fixed h=20 &mdash; replan 5&times; more</td><td>25/76 &nbsp;33%</td><td>19.3</td><td>20</td><td class="ns">p = 0.061</td></tr>
    </tbody>
  </table></div>
  <div class="wrap"><p class="note">Not one comparison is significant. An oracle handed
  ground-truth reversibility does no better than committing blindly to 100 steps.</p></div>
</section>

<section class="wrap stack">
  <p class="eyebrow">The conclusion</p>
  <h2>Right signal, wrong actuator</h2>
  <div class="card stack-sm">
    <p style="font-size:1.15rem"><b>Reversibility is measurable, predictable, and useless as a
    horizon controller on this task.</b></p>
    <p style="color:var(--ink-2)">Success is flat in horizon everywhere above the collapse threshold,
    so there is no gain available from moving within it. If low reversibility should trigger anything,
    it isn&rsquo;t &ldquo;replan more often&rdquo; &mdash; it would have to be a different recovery
    behaviour entirely.</p>
  </div>
</section>

<section class="wrap stack">
  <p class="eyebrow">How we got there</p>
  <h2>Four answers measurement overturned</h2>
  <ul>
    <li><b>&ldquo;The estimator just needs to be more accurate.&rdquo;</b> Accuracy went
    0.479 &rarr; 0.660 &rarr; 0.693 across three datasets. Control performance never moved. Ruled out
    by measuring the labels&rsquo; own repeatability at <span class="num">0.977</span> &mdash;
    measurement was never the limit.</li>
    <li><b>&ldquo;It&rsquo;s miscalibrated, so calibrate it.&rdquo;</b> Isotonic calibration made
    held-out error <span class="num">37%</span> worse. The bias doesn&rsquo;t transfer between
    episode sets.</li>
    <li><b>&ldquo;Given true reversibility, gating wins.&rdquo;</b> It did &mdash;
    <span class="num">77%</span> vs <span class="num">65%</span>,
    <span class="num">p&nbsp;=&nbsp;0.013</span>. Then we found the disturbances were firing
    <em>before the cube was grasped</em>: first contact happens at step 154 and we were perturbing
    between 40 and 160. Under corrected disturbances the same oracle <em>loses</em>.</li>
    <li><b>&ldquo;Success is monotone in how long you commit.&rdquo;</b> Our own claim, from six
    scattered conditions at <span class="num">&rho;&nbsp;=&nbsp;0.94</span>. A proper seven-point
    sweep shows a threshold, not a ramp &mdash; and the threshold is what kills the method.</li>
  </ul>
  <p class="note warn"><b>None of these failed loudly.</b> Half the disturbances did nothing while
  reporting success. A state restore silently dropped the solver&rsquo;s warm-start cache. An episode
  step counter survived a reset, so branch&nbsp;0 ran 280 steps and branch&nbsp;1 ran <em>one</em>. A
  directory flag was ignored by worker processes and quietly mixed two datasets. Every one produced
  plausible numbers, and every one was caught by a check that made the pipeline <b>refuse to
  proceed</b> &mdash; never by noticing a wrong result.</p>
</section>

<section class="wrap stack">
  <p class="eyebrow">Limits</p>
  <h2>What this does not show</h2>
  <ul>
    <li>One task, one policy, one simulator. Whether the plateau exists for other chunked policies is untested.</li>
    <li>Outside the cliff comparisons, nothing survives correction for multiple testing. Detecting the
    differences in play needs roughly 300&ndash;1250 episodes per condition; we ran 37&ndash;79.</li>
    <li>Two of the four perturbation kinds are ineffective here: blacking out 90% of the camera costs
    this policy 5 points, because it runs mostly on proprioception.</li>
    <li>An oracle reads labels measured on the very episodes it runs. It bounds what is achievable; it
    is not itself achievable.</li>
  </ul>
</section>

<footer class="wrap">
  <p>ALOHA transfer-cube in MuJoCo &middot; ACT via lerobot 0.3.2, reproduced against the reference
  evaluation seed-for-seed (48/50) &middot; 1,990 labelled states from ~2.6M simulated steps, all on a
  laptop CPU &middot; every number here is measured, including the ones that overturned an earlier claim.</p>
</footer>
'''
out = pathlib.Path("site/index.html")
out.write_text(html)
print(f"written {out.stat().st_size/1e6:.2f} MB")
