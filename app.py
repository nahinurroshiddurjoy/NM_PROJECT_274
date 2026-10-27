"""Numerical Methods Studio: interactive Streamlit application.
Run: streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import streamlit as st
from nm_core import (X, make_function, differentiate, integrate, solve_ode,
                     solve_bvp, heat_equation, monte_carlo, random_walk,
                     eigenvalues, power_method)

st.set_page_config(page_title="Numerical Methods Studio", layout="wide", initial_sidebar_state="expanded")

THEMES = {
    "Dark": {"bg":"#0b1020", "card":"#121a2b", "text":"#ecf2ff", "muted":"#9aa9c5", "border":"#2a3954", "accent":"#6f8dff", "plot":"plotly_dark"},
    "Light": {"bg":"#f4f7fc", "card":"#ffffff", "text":"#18243c", "muted":"#60718d", "border":"#dde5f0", "accent":"#476cf5", "plot":"plotly_white"}
}
TOPICS = ["Dashboard", "Differentiation", "Integration", "ODE / IVP", "BVP", "Heat PDE", "Monte Carlo", "Eigenvalues", "Learn"]
LESSONS = {
    "Differentiation": ("Finite differences approximate derivatives from nearby values.", [r"D_+f(x)=\frac{f(x+h)-f(x)}{h}",r"D_-f(x)=\frac{f(x)-f(x-h)}{h}",r"D_0f(x)=\frac{f(x+h)-f(x-h)}{2h}"], "Try x**3 at x=2, h=0.1, then reduce h."),
    "Integration": ("Divide an interval into strips and take a weighted sum of samples.", [r"I\approx h\,c\sum_{i=0}^{n}w_if(x_i)",r"h=\frac{b-a}{n}"], "Integrate x**2 from 0 to 1 with n=6."),
    "ODE / IVP": ("An initial-value problem starts at y(x0)=y0 and progresses by approximating slopes.", [r"y_{n+1}=y_n+hf(x_n,y_n)",r"y_{n+1}=y_n+\frac{k_1+2k_2+2k_3+k_4}{6}"], "Compare Euler and RK4 for y'=x+y."),
    "BVP": ("Boundary-value problems specify y at both endpoints.", [r"y''=p(x)y'+q(x)y+r(x)",r"y''(x_i)\approx\frac{y_{i-1}-2y_i+y_{i+1}}{h^2}"], "Try y''=-2 and y(0)=y(1)=0."),
    "Heat PDE": ("The explicit finite-difference method models heat diffusion along a rod.", [r"u_t=\alpha u_{xx}",r"u_i^{k+1}=u_i^k+\lambda(u_{i-1}^k-2u_i^k+u_{i+1}^k)",r"\lambda=\frac{\alpha\Delta t}{\Delta x^2}\leq\frac12"], "Observe how the sine-shaped temperature profile decays."),
    "Monte Carlo": ("Random sampling estimates integrals and simulates uncertain paths.", [r"I\approx(b-a)\frac{1}{N}\sum_{j=1}^Nf(U_j)",r"S_n=\sum_{j=1}^n\xi_j"], "Increase samples and compare accuracy."),
    "Eigenvalues": ("An eigenvector is a direction stretched by a matrix.", [r"Av=\lambda v",r"\lambda\approx\frac{v^TAv}{v^Tv}"], "Compare direct eigenvalues to power iteration."),
}

def style(theme):
    t=THEMES[theme]
    st.markdown("""<style>
    .stApp {background:BG; color:TXT;}
    [data-testid="stSidebar"]{background:SIDE; border-right:1px solid BORDER;}
    [data-testid="stMetric"] {background:CARD;border:1px solid BORDER;border-radius:14px;padding:16px;}
    div[data-testid="stForm"]{background:CARD;border:1px solid BORDER;border-radius:16px;padding:18px;}
    div[data-testid="stTabs"] button{font-weight:650;}
    h1,h2,h3{letter-spacing:-.035em;}
    .hero{padding:24px 28px;border:1px solid BORDER;border-radius:20px;background:linear-gradient(125deg,CARD,ACCENT1);margin-bottom:20px;}
    .kicker{color:ACCENT; font-size:12px;font-weight:800;letter-spacing:2px;text-transform:uppercase;}
    .hero h1{font-size:36px;margin:7px 0;color:TXT;}
    .hero p{color:MUTED;margin-bottom:0;font-size:16px;}
    .small-note{font-size:13px;color:MUTED;}
    .step-box{padding:10px 14px;border-left:3px solid ACCENT;background:CARD;border-radius:5px;margin:10px 0;}
    </style>""".replace("BG",t["bg"]).replace("SIDE",t["card"]).replace("TXT",t["text"]).replace("BORDER",t["border"]).replace("CARD",t["card"]).replace("ACCENT1", "#24355b" if theme=="Dark" else "#e6edff").replace("ACCENT",t["accent"]).replace("MUTED",t["muted"]),unsafe_allow_html=True)

def hero(title, subtitle, tag="NUMERICAL METHODS STUDIO"):
    st.markdown(f'<div class="hero"><span class="kicker">{tag}</span><h1>{title}</h1><p>{subtitle}</p></div>',unsafe_allow_html=True)

def graph(fig, title):
    fig.update_layout(title=dict(text=title,font=dict(size=18)), template=THEMES[theme]["plot"],
        height=440, margin=dict(l=36,r=20,t=65,b=35), hovermode="x unified",
        legend=dict(orientation="h",y=1.13,x=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=True,gridcolor="rgba(130,145,170,.15)",zeroline=False)
    fig.update_yaxes(showgrid=True,gridcolor="rgba(130,145,170,.15)",zeroline=False)
    st.plotly_chart(fig,use_container_width=True)

def table(data,name):
    df=pd.DataFrame(data)
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.download_button("Download CSV table",df.to_csv(index=False).encode(),file_name=name,mime="text/csv")

def result_for(key, submitted, callback):
    if submitted:
        try:
            st.session_state[key]=callback()
        except Exception as exc:
            st.session_state.pop(key,None)
            st.error(f"Calculation could not be completed: {exc}")
    return st.session_state.get(key)

def metrics(pairs):
    cols=st.columns(len(pairs))
    for c,(label,value) in zip(cols,pairs):
        c.metric(label, f"{value:.8g}" if isinstance(value,(int,float,np.floating)) else str(value))

def steps(content):
    for idx,txt in enumerate(content,1):
        st.markdown(f'<div class="step-box"><b>Step {idx}.</b> {txt}</div>',unsafe_allow_html=True)

def lesson(topic):
    desc,forms,exercise=LESSONS[topic]
    st.write(desc)
    for fm in forms: st.latex(fm)
    st.info("Practice: "+exercise)

with st.sidebar:
    st.markdown("""<div style="display:flex;align-items:center;gap:11px;padding:8px 0 14px"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="2" y="2" width="20" height="20" rx="5"/><path d="M6 16l4-9 4 10 4-7"/></svg><span style="font-size:21px;font-weight:800;letter-spacing:-.7px">NM / Studio</span></div>""",unsafe_allow_html=True)
    st.caption("Interactive learning & calculation")
    page=st.radio("MODULES", TOPICS)
    st.divider()
    theme=st.selectbox("Appearance",["Dark","Light"])
    st.caption("Functions: x**2, sin(x), exp(x), sqrt(x), pi. Use y in ODE equations.")
    st.caption("No emojis. Graphs powered by Plotly.")
style(theme)

if page=="Dashboard":
    hero("Numerical Methods Studio", "Calculate, visualize, inspect numerical steps, and build understanding across seven core topics.","INTERACTIVE MATHEMATICS WORKSPACE")
    metrics([("Numerical topics",7),("Methods & simulations",18),("Visualizations","Interactive")])
    st.subheader("Choose a module")
    st.markdown("**Differentiation** — Forward, backward, central.  |  **Integration** — Trapezoid, Simpson, Weddle.  |  **ODE** — Euler, Heun, RK2, RK4.")
    st.markdown("**BVP** — Finite difference and shooting.  |  **PDE** — Heat equation.  |  **Monte Carlo** — Sampling and random walk.  |  **Eigenvalues** — Direct and power iteration.")
    st.info("Select a module in the sidebar. Edit inputs, press Calculate, then explore the Results, Graph, Calculation Table and Learn tabs.")
    st.markdown("### Recommended first experiment")
    st.latex(r"f(x)=x^3-2x+4,\quad x_0=2,\quad h=0.05")
    st.write("Start with differentiation. The exact derivative at x=2 is 10. Compare the numerical estimates.")

elif page=="Learn":
    hero("Master Numerical Methods & Streamlit","A practical learning sequence from basic calculations to a polished deployed app.")
    roadmap=[("1 / Python fundamentals","Functions, loops, NumPy, float errors, function plotting"),("2 / Streamlit foundations","st.form, inputs, state, st.columns and st.tabs"),("3 / Calculus methods","Finite differences, trapezoidal, Simpson, Weddle"),("4 / Differential equations","Euler, Heun, Runge-Kutta, BVP and heat PDE"),("5 / Simulation and linear algebra","Monte Carlo, random walk and eigenvalue methods"),("6 / Web-app delivery","Plotly, dataframes, CSV export, tests and cloud deployment")]
    for i,(name,details) in enumerate(roadmap):
        with st.expander(name,expanded=i==0):
            st.write(details)
            st.checkbox("Completed",key=f"course_{i}")
    done=sum(st.session_state.get(f"course_{i}",False) for i in range(6))
    st.progress(done/6)
    st.caption(f"{done} / 6 modules completed in the current browser session")
    st.code('''import streamlit as st
st.title("My first Streamlit calculator")
x = st.number_input("x", value=2.0)
if st.button("Calculate"):
    st.metric("x squared", x**2)''',language="python")

else:
    hero({"ODE / IVP":"Initial Value Problems", "BVP":"Boundary Value Problems", "Heat PDE":"Heat Equation PDE"}.get(page,page),"Interactive solution, formula explanations, graphical results and numerical tabulation.")
    with st.form(f"input_{page}"):
        if page=="Differentiation":
            a,b,c=st.columns(3)
            expr=a.text_input("f(x)","x**3 - 2*x + 4")
            x0=b.number_input("Evaluation x0",value=2.0)
            h=c.number_input("Step h",value=0.05,min_value=0.000001,format="%.6f")
            job=lambda: differentiate(expr,x0,h)
        elif page=="Integration":
            expr=st.text_input("f(x)","sin(x)+2")
            a,b,c,d=st.columns(4)
            left=a.number_input("Lower a",value=0.0)
            right=b.number_input("Upper b",value=3.14159265)
            n=c.number_input("Intervals n",min_value=1,max_value=10000,value=6)
            method=d.selectbox("Integration rule",["Trapezoidal","Simpson 1/3","Simpson 3/8","Weddle"])
            job=lambda:integrate(expr,left,right,n,method)
        elif page=="ODE / IVP":
            expr=st.text_input("dy/dx = f(x,y)","x+y")
            a,b,c=st.columns(3)
            x0=a.number_input("Initial x",value=0.0)
            y0=b.number_input("Initial y",value=1.0)
            target=c.number_input("Target x",value=1.0)
            a,b=st.columns(2)
            n=a.number_input("Steps",min_value=1,max_value=20000,value=10)
            method=b.selectbox("Method",["Euler","Heun","RK2 Midpoint","RK4"])
            job=lambda:solve_ode(expr,x0,y0,target,n,method)
        elif page=="BVP":
            a,b,c=st.columns(3)
            p=a.text_input("p(x) in y'' = p*y' + q*y + r",value="0")
            q=b.text_input("q(x)",value="0")
            r=c.text_input("r(x)",value="-2")
            a,b,c,d=st.columns(4)
            left=a.number_input("Left x",value=0.0)
            right=b.number_input("Right x",value=1.0)
            ya=c.number_input("Left y",value=0.0)
            yb=d.number_input("Right y",value=0.0)
            a,b=st.columns(2)
            n=a.number_input("Interior points",min_value=2,max_value=400,value=19)
            method=b.selectbox("Solver",["Finite Difference","Shooting (RK4)"])
            job=lambda:solve_bvp(p,q,r,left,right,ya,yb,n,method)
        elif page=="Heat PDE":
            a,b,c,d=st.columns(4)
            length=a.number_input("Rod length",value=1.0,min_value=0.001)
            alpha=b.number_input("Diffusivity",value=0.1,min_value=0.00001,format="%.5f")
            T=c.number_input("Final time",value=0.2,min_value=0.0001)
            nx=d.number_input("Spatial intervals",min_value=5,max_value=100,value=25)
            job=lambda:heat_equation(length,alpha,T,nx)
        elif page=="Monte Carlo":
            mode=st.selectbox("Simulation",["Integral","Random Walk"])
            if mode=="Integral":
                expr=st.text_input("f(x)","x**2")
                a,b,c=st.columns(3)
                left=a.number_input("Lower a",value=0.0)
                right=b.number_input("Upper b",value=1.0)
                n=c.number_input("Samples",min_value=100,max_value=200000,value=10000,step=100)
            else:
                a,b=st.columns(2)
                n=a.number_input("Steps / walk",min_value=1,max_value=5000,value=200)
                paths=b.number_input("Paths",min_value=1,max_value=100,value=10)
            seed=st.number_input("Random seed",min_value=0,max_value=999999,value=42)
            job=(lambda:monte_carlo(expr,left,right,n,seed)) if mode=="Integral" else (lambda:random_walk(n,paths,seed))
        elif page=="Eigenvalues":
            size=st.selectbox("Matrix dimension",[2,3])
            default=[[4.,1.],[1.,3.]] if size==2 else [[4.,1.,0.],[1.,3.,1.],[0.,1.,2.]]
            st.caption("Edit the matrix numbers below")
            matrix=st.data_editor(pd.DataFrame(default),use_container_width=True,hide_index=True,key=f"matrix{size}")
            method=st.selectbox("Algorithm",["Direct","Power Method"])
            job=(lambda:eigenvalues(matrix.to_numpy())) if method=="Direct" else (lambda:power_method(matrix.to_numpy()))
        submitted=st.form_submit_button("Calculate / Solve",type="primary",use_container_width=True)
    # Inputs can change without submitting; results belong to the last solved values.
    state_key=f"result_{page}"
    result=result_for(state_key,submitted,job)
    if result is None:
        st.info("Enter your values and select Calculate / Solve to generate an interactive graph and calculation table.")
    else:
        overview,visual,tabulation,explanation=st.tabs(["Results", "Interactive Graph", "Calculation Table", "Method & Learning"])
        with overview:
            if page=="Differentiation":
                metrics([("Forward",result["results"]["Forward"]),("Backward",result["results"]["Backward"]),("Central",result["results"]["Central"]),("Exact",result["exact"])])
                st.latex(r"f'(x)="+sp.latex(result["exact_expression"]))
                steps(["Evaluate f(x0-h), f(x0) and f(x0+h).","Divide differences by h (or 2h for central).","Compare numerical approximations with the analytical derivative."])
            elif page=="Integration":
                metrics([("Estimate",result["estimate"]),("Reference",result["reference"]),("Abs error",result["error"]),("h",result["h"])])
                steps(["Partition [a,b] into n subintervals.","Evaluate f(x) at nodes and apply the selected weights.","Sum weight × f(x), then multiply by h and the rule factor."])
            elif page=="ODE / IVP":
                metrics([("Numerical y(target)",result["result"]),("Step h",result["h"]),("Reference",result["reference"] if result["reference"] is not None else "N/A")])
                steps(["Set starting point (x0,y0) and step h.","Calculate the slopes (k1...k4 as required).","Update y each step to obtain the final result."])
            elif page=="BVP":
                metrics([("Mesh spacing",result["h"]),("Boundary residual",result["residual"]),("Points",len(result["x"]))])
                steps(["Define coefficients p(x), q(x), r(x) and endpoint values.","Create the mesh and solve via finite differences or shooting.","Check the right boundary and inspect the solution curve."])
            elif page=="Heat PDE":
                metrics([("Time steps",result["steps"]),("Stability lambda",result["ratio"]),("Max error",result["max_error"]),("dt",result["dt"])])
                st.success("Stable explicit finite-difference simulation: lambda <= 0.5")
                steps(["Create evenly spaced points along the rod.","Choose a stable time increment.","Apply the finite-difference heat update repeatedly."])
            elif page=="Monte Carlo":
                if "estimate" in result:
                    metrics([("Estimate",result["estimate"]),("Reference",result["reference"]),("Absolute error",result["error"])])
                    st.write("Approximate 95% confidence interval:",result["confidence"])
                else:
                    metrics([("Mean endpoint",result["mean"]),("Variance",result["variance"]),("Theory variance",result["theoretical_variance"])])
            elif page=="Eigenvalues":
                if "values" in result:
                    st.write("Eigenvalues:",np.round(result["values"],8)); st.write("Eigenvectors (columns):",np.round(result["vectors"],8))
                else:
                    metrics([("Dominant eigenvalue",result["value"]),("Residual",result["residual"]),("Converged",str(result["converged"]))])
                    st.write("Eigenvector:",np.round(result["vector"],8))
        with visual:
            fig=go.Figure()
            if page=="Differentiation":
                f=make_function(str(result["expression"]))[1]
                xs=np.linspace(result["x0"]-2,result["x0"]+2,250)
                ys=np.asarray(f(xs),dtype=float)
                if ys.ndim==0: ys=np.full_like(xs,float(ys))
                fig.add_trace(go.Scatter(x=xs,y=ys,name="f(x)",mode="lines"))
                fig.add_trace(go.Scatter(x=xs,y=result["f_zero"]+result["exact"]*(xs-result["x0"]),name="Tangent",line=dict(dash="dash")))
                fig.add_trace(go.Scatter(x=[result["x0"]],y=[result["f_zero"]],name="Evaluation point",mode="markers",marker=dict(size=11)))
                graph(fig,"Function and analytical tangent")
                err=go.Figure()
                err.add_trace(go.Scatter(x=[row["h"] for row in result["convergence"]],y=[row["Absolute error"] for row in result["convergence"]],mode="lines+markers",name="Central error"))
                err.update_xaxes(type="log",title="Step h");err.update_yaxes(title="Absolute error")
                graph(err,"Convergence as h changes")
            elif page=="Integration":
                fig.add_trace(go.Scatter(x=result["x"],y=result["y"],mode="lines+markers",fill="tozeroy",name="Sampled f(x)"))
                graph(fig,"Area and integration nodes")
            elif page=="ODE / IVP":
                fig.add_trace(go.Scatter(x=[r["x"] for r in result["rows"]],y=[r["y"] for r in result["rows"]],mode="lines+markers",name="ODE solution"))
                graph(fig,"Numerical ODE trajectory")
            elif page=="BVP":
                fig.add_trace(go.Scatter(x=result["x"],y=result["y"],mode="lines+markers",name="BVP solution"))
                graph(fig,"Boundary-value solution")
            elif page=="Heat PDE":
                for ti,ys in result["snapshots"]:
                    fig.add_trace(go.Scatter(x=result["x"],y=ys,mode="lines",name=f"Time {ti:.3f}"))
                fig.add_trace(go.Scatter(x=result["x"],y=result["exact"],line=dict(dash="dash"),name="Exact at final time"))
                graph(fig,"Temperature versus position")
            elif page=="Monte Carlo":
                if "estimate" in result:
                    fig.add_trace(go.Histogram(x=result["sample_values"],nbinsx=35,name="f(random x)"))
                    graph(fig,"Sampled values distribution")
                else:
                    for i,track in enumerate(result["tracks"]):
                        fig.add_trace(go.Scatter(y=track,mode="lines",name=f"Path {i+1}"))
                    graph(fig,"Random walk trajectories")
            elif page=="Eigenvalues":
                if "history" in result:
                    fig.add_trace(go.Scatter(x=[r["Iteration"] for r in result["history"]],y=[max(r["Residual"],1e-16) for r in result["history"]],mode="lines+markers",name="Residual"))
                    fig.update_yaxes(type="log")
                    graph(fig,"Power iteration convergence")
                else:
                    vals=result["values"]
                    fig.add_trace(go.Scatter(x=np.real(vals),y=np.imag(vals),mode="markers+text",text=[f"λ{i+1}" for i in range(len(vals))],textposition="top center",marker=dict(size=14),name="Eigenvalues"))
                    fig.update_xaxes(title="Real part");fig.update_yaxes(title="Imaginary part")
                    graph(fig,"Eigenvalues in the complex plane")
        with tabulation:
            if page=="Differentiation":
                table([{"Method":m,"Estimate":v,"Absolute error":abs(v-result["exact"])} for m,v in result["results"].items()],"differentiation.csv")
                st.markdown("#### Step-size convergence")
                table(result["convergence"],"convergence.csv")
            elif page=="Integration": table(result["rows"],"integration.csv")
            elif page=="ODE / IVP": table(result["rows"],"ode.csv")
            elif page=="BVP": table([{"x":x,"y":y} for x,y in zip(result["x"],result["y"])],"bvp.csv")
            elif page=="Heat PDE": table([{"x":x,"Numerical u":y,"Exact u":ex,"Abs error":abs(y-ex)} for x,y,ex in zip(result["x"],result["y"],result["exact"])],"heat.csv")
            elif page=="Monte Carlo":
                if "estimate" in result:table([{"Sample x":x,"f(x)":y} for x,y in zip(result["samples"][:500],result["sample_values"][:500])],"monte_carlo_first_500.csv")
                else:table([{"Step":i,**{f"Walk {j+1}":int(path[i]) for j,path in enumerate(result["tracks"][:,:])}} for i in range(min(501,result["tracks"].shape[1]))],"random_walk.csv")
            elif page=="Eigenvalues":
                if "history" in result:table(result["history"],"power_iterations.csv")
                else:table([{"Eigenpair":i+1,"Eigenvalue":str(v),"Eigenvector":str(result["vectors"][:,i])} for i,v in enumerate(result["values"])],"eigenpairs.csv")
        with explanation:lesson(page)

st.divider()
st.caption("Numerical Methods Studio · Interactive graphs · Worked calculation tables · Educational numerical estimates")
