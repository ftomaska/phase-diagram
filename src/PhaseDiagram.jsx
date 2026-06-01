import { useState, useRef, useCallback, useEffect } from "react";

// ── constants ──────────────────────────────────────────────────────────────
// X axis: FOV area in mm²
// Calibration default: 700 ph/s at 0.5625 mm² (750×750 µm)
const DEFAULT_REF_FLUX = 700;          // ph/s per cell
const DEFAULT_REF_AREA = 0.5625;       // mm²  (750 µm × 750 µm)

// Pixel budget: assume resonant scanner, fixed pixel rate
// 512 lines × frame_rate = lines/s; line_spacing = sqrt(area)/512
// In Wilt et al. style: line spacing (µm) = sqrt(FOV_area_µm²) / N_lines
// We express FOV in mm², so side_µm = 1000*sqrt(area_mm²)
// line_spacing_µm = 1000*sqrt(area_mm²) / N_LINES
const N_LINES = 512;

// Indicators from literature
// GCaMP6s/6f: Chen et al. 2013; jGCaMP8: Zhang et al. 2023 table
const INDICATORS = [
  { label: "GCaMP6s",  dff: 0.25,  tau: 0.500, color: "#4ec9b0" },
  { label: "GCaMP6f",  dff: 0.12,  tau: 0.150, color: "#79c0ff" },
  { label: "jGCaMP8s", dff: 1.10,  tau: 0.1883, color: "#ffa657" },
  { label: "jGCaMP8m", dff: 0.75,  tau: 0.0380, color: "#ff7b72" },
  { label: "jGCaMP8f", dff: 0.37,  tau: 0.0193, color: "#d2a8ff" },
  { label: "Custom",   dff: 0.25,  tau: 0.500, color: "#e6edf3" },
];

const N_FOV = 140;
const N_FR  = 100;
// FOV area in mm²
const AREA_MIN = 0.01, AREA_MAX = 25;   // mm²
const FR_MIN   = 1,    FR_MAX   = 100;  // Hz

function logspace(mn, mx, n) {
  return Array.from({length:n},(_,i) => mn * Math.pow(mx/mn, i/(n-1)));
}
const AREA_VALS = logspace(AREA_MIN, AREA_MAX, N_FOV);
const FR_VALS   = logspace(FR_MIN,   FR_MAX,   N_FR);

// F0 scales inversely with FOV area
function calcF0(area_mm2, refFlux, refArea_mm2) {
  return refFlux * refArea_mm2 / area_mm2;
}

// d′ formula (Wilt et al. 2013)
function dprime(area_mm2, dff, tau, refFlux, refArea) {
  const F0 = calcF0(area_mm2, refFlux, refArea);
  return dff * Math.sqrt(0.5 * F0 * tau);
}

// Line spacing in µm for a given FOV area (mm²) and frame rate
// side_µm = 1000*sqrt(area_mm²); line_spacing = side_µm / N_LINES
// In Wilt et al. the line spacing contours are diagonal because
// with a fixed pixel clock: N_lines = pixel_rate / (pixels_per_line * frame_rate)
// Here we use fixed N_LINES=512 (simpler, matches figure style)
function lineSpacingMicron(area_mm2) {
  return 1000 * Math.sqrt(area_mm2) / N_LINES;
}

function nyquistHz(tau) { return 2 / tau; }

// ── viridis colormap ───────────────────────────────────────────────────────
function dpToColor(dp) {
  const t = Math.min(Math.max(dp / 10, 0), 1);
  const stops = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  const seg = t*(stops.length-1);
  const i   = Math.min(Math.floor(seg), stops.length-2);
  const f   = seg-i;
  const c   = stops[i].map((v,k)=>Math.round(v+f*(stops[i+1][k]-v)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ── raster (d′ independent of FR so FR axis is just for display) ───────────
function buildRaster(dff, tau, refFlux, refArea) {
  return AREA_VALS.map(area =>
    FR_VALS.map(() => dprime(area, dff, tau, refFlux, refArea))
  );
}

// ── canvas renderer ────────────────────────────────────────────────────────
function renderCanvas(canvas, raster, dff, tau, refFlux, refArea, point, hovering, showLS) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W   = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);

  // heatmap
  const cw = W/N_FOV, ch = H/N_FR;
  for (let xi=0;xi<N_FOV;xi++)
    for (let yi=0;yi<N_FR;yi++) {
      ctx.fillStyle = dpToColor(raster[xi][N_FR-1-yi]);
      ctx.fillRect(xi*cw, yi*ch, cw+1, ch+1);
    }

  // coord helpers
  const areaToX = a =>
    ((Math.log(a)-Math.log(AREA_MIN))/(Math.log(AREA_MAX)-Math.log(AREA_MIN)))*W;
  const frToY = fr =>
    H-((Math.log(fr)-Math.log(FR_MIN))/(Math.log(FR_MAX)-Math.log(FR_MIN)))*H;

  // ── diagonal line-spacing contours (Wilt et al. style) ──────────────────
  // line_spacing = 1000*sqrt(area)/512  → area = (ls*512/1000)²
  // These are vertical lines (FR-independent) for fixed N_LINES
  // To get the diagonal look we instead draw iso-ls lines that are
  // *diagonal* by tying ls to frame rate via a fixed pixel clock:
  // pixel_clock = N_LINES * pixels_per_line * FR
  // pixels_per_line = N_LINES (square), so pixel_clock = N_LINES² * FR
  // actual N_lines per frame = pixel_clock / (FR * pixels_per_line) = N_LINES (always 512)
  // → line spacing is still purely area-dependent.
  // BUT in the Wilt fig the scanner has a fixed total pixel rate P_tot:
  // N_lines(fr) = P_tot / (fr * pixels_per_line)
  // line_spacing(area,fr) = 1000*sqrt(area) / N_lines(fr) = 1000*sqrt(area)*fr / (P_tot/pixels_per_line)
  // i.e. ls ∝ sqrt(area) * fr → diagonal contours on log-log axes
  // We use P_tot = N_LINES² * FR_REF where FR_REF = 30 Hz (typical resonant scanner)
  const FR_REF     = 30;                          // Hz reference frame rate
  const PIXEL_RATE = N_LINES * N_LINES * FR_REF;  // total pixels/s

  if (showLS) {
    const lsLevels = [0.1, 0.2, 0.5, 1, 2, 5, 10];  // µm
    lsLevels.forEach(ls => {
      // ls = 1000*sqrt(area)*fr / (PIXEL_RATE/N_LINES)
      // → for each fr, area = (ls * PIXEL_RATE / (N_LINES * fr * 1000))²
      ctx.beginPath();
      const isBound = ls === 0.5 || ls === 5;
      ctx.strokeStyle = isBound ? "rgba(255,200,50,0.85)" : "rgba(255,255,255,0.45)";
      ctx.lineWidth   = isBound ? 2 : 1;
      ctx.setLineDash(isBound ? [8,4] : [5,4]);

      let started = false;
      for (let yi=0; yi<N_FR; yi++) {
        const fr   = FR_VALS[N_FR-1-yi];
        const area = Math.pow(ls * PIXEL_RATE / (N_LINES * fr * 1000), 2);
        const x    = areaToX(area);
        const y    = frToY(fr);
        if (area < AREA_MIN || area > AREA_MAX) { started=false; continue; }
        started ? ctx.lineTo(x,y) : (ctx.moveTo(x,y), started=true);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // label near top
      const fr_label = FR_MAX * 0.7;
      const area_label = Math.pow(ls * PIXEL_RATE / (N_LINES * fr_label * 1000), 2);
      if (area_label > AREA_MIN && area_label < AREA_MAX) {
        const lx = areaToX(area_label), ly = frToY(fr_label);
        ctx.save();
        ctx.translate(lx, ly);
        // slope on log-log: Δlog(fr)/Δlog(area) = -0.5 → angle ≈ -27°
        ctx.rotate(-0.42);
        ctx.fillStyle = isBound ? "rgba(255,210,60,0.95)" : "rgba(255,255,255,0.75)";
        ctx.font = `${isBound?'bold ':' '}9px Arial,sans-serif`;
        ctx.fillText(`${ls} µm`, 2, -3);
        ctx.restore();
      }
    });
  }

  // ── d′ contours (vertical — FR-independent) ────────────────────────────
  const contourLevels = [0.5, 1, 2, 3, 5, 10];
  contourLevels.forEach(level => {
    // area = refArea * refFlux * dff² * tau / (2*level²)
    const critArea = refArea * refFlux * dff * dff * tau * 0.5 / (level * level);
    const cx = areaToX(critArea);
    if (cx < 0 || cx > W) return;
    ctx.beginPath();
    ctx.strokeStyle = level===3 ? "rgba(255,70,70,0.95)" : "rgba(255,255,255,0.6)";
    ctx.lineWidth   = level===3 ? 2.5 : 1.2;
    ctx.setLineDash(level===3 ? [] : [6,4]);
    ctx.moveTo(cx,0); ctx.lineTo(cx,H); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = level===3 ? "rgba(255,100,100,0.95)" : "rgba(255,255,255,0.85)";
    ctx.font = `bold ${level===3?12:10}px Arial,sans-serif`;
    ctx.fillText(`${level}`, cx+3, 14);
  });
  // d′ label
  ctx.fillStyle="rgba(255,255,255,0.5)"; ctx.font="9px Arial,sans-serif";
  ctx.fillText("d′=",3,14);

  // ── temporal Nyquist ───────────────────────────────────────────────────
  const nyq  = nyquistHz(tau);
  const nyqY = frToY(nyq);
  if (nyqY>=0 && nyqY<=H) {
    ctx.strokeStyle="rgba(255,140,0,0.9)";
    ctx.lineWidth=2; ctx.setLineDash([10,4]);
    ctx.beginPath(); ctx.moveTo(0,nyqY); ctx.lineTo(W,nyqY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle="rgba(255,165,50,0.95)";
    ctx.font="10px Arial,sans-serif";
    ctx.fillText(`↓ Nyquist limit (${nyq.toFixed(1)} Hz)`,6,nyqY-4);
  }

  // ── draggable point ────────────────────────────────────────────────────
  if (point) {
    const px  = areaToX(point.area);
    const py  = frToY(point.fr);
    const dp  = dprime(point.area, dff, tau, refFlux, refArea);
    const ls  = showLS ? lineSpacingMicron(point.area) : null;
    const warn = dp<3 || point.fr<nyq || (showLS && (ls<0.5 || ls>5));

    const grd = ctx.createRadialGradient(px,py,0,px,py,hovering?30:22);
    grd.addColorStop(0, warn?"rgba(255,60,60,0.45)":"rgba(255,255,255,0.3)");
    grd.addColorStop(1,"rgba(0,0,0,0)");
    ctx.fillStyle=grd;
    ctx.beginPath(); ctx.arc(px,py,hovering?30:22,0,Math.PI*2); ctx.fill();

    ctx.beginPath(); ctx.arc(px,py,hovering?13:10,0,Math.PI*2);
    ctx.strokeStyle=warn?"#ff4040":"#ffffff"; ctx.lineWidth=2.5; ctx.stroke();
    ctx.beginPath(); ctx.arc(px,py,5,0,Math.PI*2);
    ctx.fillStyle=warn?"#ff4040":"#ffffff"; ctx.fill();
  }
}

// ── SVG axis overlay ───────────────────────────────────────────────────────
function AxesLabels({ W, H, pad }) {
  const areaTicks = [0.01,0.05,0.1,0.25,0.5,1,2,5,10,25];
  const frTicks   = [1,2,5,10,30,100];

  const areaToX = a =>
    pad.l+((Math.log(a)-Math.log(AREA_MIN))/(Math.log(AREA_MAX)-Math.log(AREA_MIN)))*(W-pad.l-pad.r);
  const frToY = fr =>
    pad.t+(1-(Math.log(fr)-Math.log(FR_MIN))/(Math.log(FR_MAX)-Math.log(FR_MIN)))*(H-pad.t-pad.b);

  return (
    <svg style={{position:"absolute",inset:0,pointerEvents:"none"}} width={W} height={H}>
      {areaTicks.map(v=>(
        <g key={v}>
          <line x1={areaToX(v)} y1={H-pad.b} x2={areaToX(v)} y2={H-pad.b+5} stroke="#8899aa" strokeWidth={1}/>
          <text x={areaToX(v)} y={H-pad.b+16} textAnchor="middle"
                fill="#8899aa" fontSize={10} fontFamily="Arial,sans-serif">
            {v<1?v:v}
          </text>
        </g>
      ))}
      {frTicks.map(v=>(
        <g key={v}>
          <line x1={pad.l-5} y1={frToY(v)} x2={pad.l} y2={frToY(v)} stroke="#8899aa" strokeWidth={1}/>
          <text x={pad.l-8} y={frToY(v)+4} textAnchor="end"
                fill="#8899aa" fontSize={10} fontFamily="Arial,sans-serif">{v}</text>
        </g>
      ))}
      <text x={(W+pad.l-pad.r)/2} y={H-1} textAnchor="middle"
            fill="#aabbcc" fontSize={12} fontFamily="Arial,sans-serif">
        FOV area (mm²)
      </text>
      <text x={10} y={H/2} textAnchor="middle"
            fill="#aabbcc" fontSize={12} fontFamily="Arial,sans-serif"
            transform={`rotate(-90,10,${H/2})`}>Frame rate (Hz)</text>
    </svg>
  );
}

// ── colorbar ───────────────────────────────────────────────────────────────
function Colorbar({ H, pad }) {
  const steps=60;
  return (
    <svg width={48} height={H} style={{flexShrink:0}}>
      {Array.from({length:steps},(_,i)=>{
        const y=pad.t+(i/steps)*(H-pad.t-pad.b);
        const h=(H-pad.t-pad.b)/steps+1;
        return <rect key={i} x={4} y={y} width={16} height={h} fill={dpToColor((1-i/steps)*10)}/>;
      })}
      {[0,2,4,6,8,10].map(v=>{
        const y=pad.t+(1-v/10)*(H-pad.t-pad.b);
        return <g key={v}>
          <line x1={20} y1={y} x2={24} y2={y} stroke="#8899aa" strokeWidth={1}/>
          <text x={27} y={y+4} fill="#aabbcc" fontSize={10} fontFamily="Arial,sans-serif">{v}</text>
        </g>;
      })}
      <text x={42} y={H/2} textAnchor="middle" fill="#aabbcc" fontSize={10}
            fontFamily="Arial,sans-serif" transform={`rotate(90,42,${H/2})`}>d′</text>
    </svg>
  );
}

// ── main ───────────────────────────────────────────────────────────────────
export default function PhaseDiagram() {
  const [indicatorIdx, setIndicatorIdx] = useState(0);
  const [dff,     setDff]     = useState(INDICATORS[0].dff);
  const [tau,     setTau]     = useState(INDICATORS[0].tau);
  const [refFlux, setRefFlux] = useState(DEFAULT_REF_FLUX);
  const [refArea, setRefArea] = useState(DEFAULT_REF_AREA);
  const [fluxInput,    setFluxInput]    = useState("700");
  const [refAreaInput, setRefAreaInput] = useState("0.5625");
  const [showLS,  setShowLS]  = useState(true);

  const [point,    setPoint]    = useState({ area: 0.5625, fr: 13 });
  const [areaInput,setAreaInput]= useState("0.5625");
  const [frInput,  setFrInput]  = useState("13");
  const [dragging, setDragging] = useState(false);
  const [hovering, setHovering] = useState(false);

  const canvasRef = useRef(null);
  const PAD = {l:44,r:8,t:10,b:34};
  const CW=580, CH=400;

  const raster = buildRaster(dff, tau, refFlux, refArea);

  const xToArea = useCallback(x=>{
    const t=Math.max(0,Math.min(1,x/CW));
    return AREA_MIN*Math.pow(AREA_MAX/AREA_MIN,t);
  },[]);
  const yToFr = useCallback(y=>{
    const t=Math.max(0,Math.min(1,1-y/CH));
    return FR_MIN*Math.pow(FR_MAX/FR_MIN,t);
  },[]);

  useEffect(()=>{
    setAreaInput(point.area.toFixed(4));
    setFrInput(point.fr.toFixed(1));
  },[point]);

  useEffect(()=>{
    renderCanvas(canvasRef.current,raster,dff,tau,refFlux,refArea,point,hovering||dragging,showLS);
  },[raster,dff,tau,refFlux,refArea,point,hovering,dragging,showLS]);

  const applyIndicator = idx=>{
    setIndicatorIdx(idx);
    if(idx<INDICATORS.length-1){setDff(INDICATORS[idx].dff);setTau(INDICATORS[idx].tau);}
  };
  const commitFlux=()=>{const v=parseFloat(fluxInput);if(!isNaN(v)&&v>0)setRefFlux(v);};
  const commitRefArea=()=>{const v=parseFloat(refAreaInput);if(!isNaN(v)&&v>0)setRefArea(v);};

  const getXY=e=>{const r=canvasRef.current.getBoundingClientRect();return{x:e.clientX-r.left,y:e.clientY-r.top};};
  const onMouseDown=e=>{const{x,y}=getXY(e);setDragging(true);setPoint({area:xToArea(x),fr:yToFr(y)});};
  const onMouseMove=e=>{if(!dragging)return;const{x,y}=getXY(e);setPoint({area:xToArea(x),fr:yToFr(y)});};
  const onMouseUp=()=>setDragging(false);

  const dp   = dprime(point.area, dff, tau, refFlux, refArea);
  const nyq  = nyquistHz(tau);
  const ls   = lineSpacingMicron(point.area);
  const F0now= calcF0(point.area, refFlux, refArea);

  const warnDp  = dp<3;
  const warnTmp = point.fr<nyq;
  const warnOv  = showLS && ls<0.5;
  const warnUn  = showLS && ls>5;
  const anyWarn = warnDp||warnTmp||warnOv||warnUn;

  return (
    <div style={{minHeight:"100vh",background:"#0d1117",display:"flex",
                 alignItems:"center",justifyContent:"center",
                 fontFamily:"Arial,sans-serif",padding:20,boxSizing:"border-box"}}>
      <div style={{width:"100%",maxWidth:980}}>

        {/* header */}
        <div style={{marginBottom:16}}>
          <div style={{color:"#4ec9b0",fontSize:10,letterSpacing:3,textTransform:"uppercase",marginBottom:3}}>
            2P Calcium Imaging
          </div>
          <h1 style={{margin:0,color:"#e6edf3",fontSize:20,fontWeight:600,letterSpacing:-0.5}}>
            Single-Spike Detection Phase Diagram
          </h1>
          <div style={{color:"#6e7681",fontSize:10,marginTop:3}}>
            d′ = ΔF/F · √(½ · F₀ · τ) &nbsp;·&nbsp; F₀ ∝ 1/FOV area &nbsp;·&nbsp; line spacing contours assume {N_LINES}² px at {FR_REF} Hz reference
          </div>
        </div>

        <div style={{display:"flex",gap:18,alignItems:"flex-start"}}>

          {/* controls */}
          <div style={{width:215,flexShrink:0}}>

            <Section title="Indicator">
              <div style={{display:"flex",flexWrap:"wrap",gap:3,marginBottom:10}}>
                {INDICATORS.map((ind,i)=>(
                  <button key={i} onClick={()=>applyIndicator(i)} style={{
                    padding:"3px 7px",fontSize:9,border:"none",borderRadius:4,
                    cursor:"pointer",fontFamily:"inherit",
                    background:indicatorIdx===i?ind.color:"#21262d",
                    color:indicatorIdx===i?"#0d1117":"#8b949e",
                    fontWeight:indicatorIdx===i?600:400,
                  }}>{ind.label}</button>
                ))}
              </div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6,marginBottom:8,
                           background:"#161b22",borderRadius:6,padding:"8px 10px"}}>
                <StatSmall label="ΔF/F" value={`${(dff*100).toFixed(0)}%`}/>
                <StatSmall label="τ" value={`${(tau*1000).toFixed(0)} ms`}/>
              </div>
              <Slider label="ΔF/F" value={dff} min={0.05} max={3} step={0.05}
                      fmt={v=>`${(v*100).toFixed(0)}%`}
                      onChange={v=>{setDff(v);setIndicatorIdx(5);}}/>
              <Slider label="τ decay" value={tau} min={0.01} max={1.5} step={0.01}
                      fmt={v=>`${(v*1000).toFixed(0)} ms`}
                      onChange={v=>{setTau(v);setIndicatorIdx(5);}}/>
            </Section>

            <Section title="Flux Calibration">
              <div style={{color:"#4e5a68",fontSize:9,marginBottom:6,lineHeight:1.5}}>
                F₀ measured at reference FOV
              </div>
              <div style={{marginBottom:6}}>
                <Label>F₀ (ph/s per cell)</Label>
                <input value={fluxInput} onChange={e=>setFluxInput(e.target.value)}
                  onBlur={commitFlux} onKeyDown={e=>e.key==="Enter"&&commitFlux()}
                  style={inputStyle}/>
              </div>
              <div>
                <Label>at FOV area (mm²)</Label>
                <input value={refAreaInput} onChange={e=>setRefAreaInput(e.target.value)}
                  onBlur={commitRefArea} onKeyDown={e=>e.key==="Enter"&&commitRefArea()}
                  style={inputStyle}/>
              </div>
            </Section>

            <Section title="Your Setup">
              <div style={{marginBottom:6}}>
                <Label>FOV area (mm²)</Label>
                <input value={areaInput} onChange={e=>setAreaInput(e.target.value)}
                  onBlur={()=>{const v=parseFloat(areaInput);if(!isNaN(v)&&v>0)
                    setPoint(p=>({...p,area:Math.max(AREA_MIN,Math.min(AREA_MAX,v))}));}}
                  style={inputStyle}/>
                <div style={{color:"#6e7681",fontSize:9,marginTop:2}}>
                  ≈ {(Math.sqrt(point.area)*1000).toFixed(0)}×{(Math.sqrt(point.area)*1000).toFixed(0)} µm
                </div>
              </div>
              <div>
                <Label>Frame rate (Hz)</Label>
                <input value={frInput} onChange={e=>setFrInput(e.target.value)}
                  onBlur={()=>{const v=parseFloat(frInput);if(!isNaN(v)&&v>0)
                    setPoint(p=>({...p,fr:Math.max(FR_MIN,Math.min(FR_MAX,v))}));}}
                  style={inputStyle}/>
              </div>
            </Section>

            <Section title="Readout">
              <Stat label="d′"           value={dp.toFixed(2)}                    warn={warnDp} good={dp>=3}/>
              <Stat label="F₀ at setup"  value={`${Math.round(F0now)} ph/s`}/>
              <Stat label="Line spacing" value={`${ls.toFixed(2)} µm/px`}         warn={warnOv||warnUn}/>
              <Stat label="Nyquist"      value={`${nyq.toFixed(1)} Hz`}           warn={warnTmp}/>
            </Section>

            <div style={{marginBottom:10}}>
              <label style={{display:"flex",alignItems:"center",gap:6,cursor:"pointer",
                             color:"#8b949e",fontSize:10}}>
                <input type="checkbox" checked={showLS} onChange={e=>setShowLS(e.target.checked)}
                       style={{accentColor:"#4ec9b0"}}/>
                Show line spacing contours
              </label>
            </div>

            {anyWarn?(
              <div style={{background:"#2d1a1a",border:"1px solid #ff4040",borderRadius:6,padding:"10px 12px"}}>
                <div style={{color:"#ff6060",fontSize:10,fontWeight:600,letterSpacing:2,marginBottom:6}}>⚠ WARNINGS</div>
                {warnDp  && <Warn>d′ &lt; 3: single-spike detection unreliable</Warn>}
                {warnTmp && <Warn>Frame rate below Nyquist for τ={`${(tau*1000).toFixed(0)}`} ms</Warn>}
                {warnOv  && <Warn>Spatially oversampled (&lt;0.5 µm/px)</Warn>}
                {warnUn  && <Warn>Spatially undersampled (&gt;5 µm/px)</Warn>}
              </div>
            ):(
              <div style={{background:"#0f2318",border:"1px solid #238636",borderRadius:6,padding:"10px 12px"}}>
                <div style={{color:"#3fb950",fontSize:10}}>✓ All parameters nominal</div>
              </div>
            )}
          </div>

          {/* canvas */}
          <div style={{flex:1}}>
            <div style={{position:"relative",display:"flex"}}>
              <div style={{position:"relative"}}>
                <canvas ref={canvasRef} width={CW} height={CH}
                  style={{display:"block",borderRadius:6,border:"1px solid #21262d",
                          cursor:dragging?"grabbing":"crosshair"}}
                  onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp}
                  onMouseLeave={()=>{setDragging(false);setHovering(false);}}
                  onMouseEnter={()=>setHovering(true)}/>
                <AxesLabels W={CW} H={CH} pad={PAD}/>
              </div>
              <Colorbar H={CH} pad={PAD}/>
            </div>
            <div style={{color:"#4e5a68",fontSize:9,marginTop:5,textAlign:"center"}}>
              Drag point · or enter values in Your Setup &nbsp;·&nbsp;
              white dashed = d′ contours &nbsp;·&nbsp; red solid = d′=3 &nbsp;·&nbsp;
              yellow dashed = 0.5 / 5 µm/px spatial bounds &nbsp;·&nbsp; orange = Nyquist
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── helpers ────────────────────────────────────────────────────────────────
function Section({title,children}){
  return <div style={{marginBottom:14}}>
    <div style={{color:"#4ec9b0",fontSize:9,letterSpacing:3,textTransform:"uppercase",marginBottom:7}}>{title}</div>
    {children}
  </div>;
}
function Label({children}){return <div style={{color:"#6e7681",fontSize:10,marginBottom:3}}>{children}</div>;}
function Slider({label,value,min,max,step,fmt,onChange}){
  return <div style={{marginBottom:9}}>
    <div style={{display:"flex",justifyContent:"space-between",marginBottom:2}}>
      <Label>{label}</Label>
      <span style={{color:"#e6edf3",fontSize:10}}>{fmt(value)}</span>
    </div>
    <input type="range" min={min} max={max} step={step} value={value}
      onChange={e=>onChange(parseFloat(e.target.value))}
      style={{width:"100%",accentColor:"#4ec9b0",cursor:"pointer"}}/>
  </div>;
}
function Stat({label,value,warn,good}){
  const color=warn?"#ff6060":good?"#3fb950":"#c9d1d9";
  return <div style={{display:"flex",justifyContent:"space-between",marginBottom:5,alignItems:"baseline"}}>
    <span style={{color:"#6e7681",fontSize:10}}>{label}</span>
    <span style={{color,fontSize:11,fontWeight:600}}>{value}</span>
  </div>;
}
function StatSmall({label,value}){
  return <div style={{textAlign:"center"}}>
    <div style={{color:"#4e5a68",fontSize:8,letterSpacing:1}}>{label}</div>
    <div style={{color:"#e6edf3",fontSize:11,fontWeight:600}}>{value}</div>
  </div>;
}
function Warn({children}){return <div style={{color:"#ff8080",fontSize:9,marginBottom:3}}>· {children}</div>;}
const inputStyle={width:"100%",background:"#161b22",border:"1px solid #30363d",
  borderRadius:4,color:"#e6edf3",fontSize:11,padding:"5px 8px",
  fontFamily:"Arial,sans-serif",boxSizing:"border-box",outline:"none"};
const FR_REF=30;
