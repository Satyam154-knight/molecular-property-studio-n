import pandas as pd
import streamlit as st
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, Draw

st.set_page_config(page_title="Molecular Property Studio", page_icon="🧪", layout="wide")

# Clean, high-contrast UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#f3f6fb;color:#17263d;}
[data-testid="stSidebar"] {background:#101a2b;}
[data-testid="stSidebar"] * {color:#edf3ff !important;}
.hero {padding:28px 32px;border-radius:20px;background:linear-gradient(120deg,#111d35,#17465a 65%,#087e83);margin-bottom:22px;}
.hero h1 {color:#fff !important;font-family:'Space Grotesk',sans-serif;font-size:2.2rem;margin:6px 0;}
.hero p {color:#e1f1f5 !important;line-height:1.6;margin:0;}
.eyebrow {color:#71e1d0 !important;font-weight:700;font-size:.75rem;letter-spacing:.15em;text-transform:uppercase;}
.section-title {font-family:'Space Grotesk',sans-serif;color:#14243b !important;font-size:1.35rem;font-weight:700;margin:16px 0;}
[data-testid="stMetric"] {background:white;border:1px solid #dfe7f0;padding:14px;border-radius:14px;}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] * {color:#52647b !important;}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] * {color:#14243b !important;}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] * {color:#52647b !important;}
.stButton button,.stDownloadButton button {border-radius:10px;font-weight:700;}
</style>
""", unsafe_allow_html=True)

MOLECULES = {
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Ethanol": "CCO",
    "Water": "O",
    "Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Nicotine": "CN1CCC[C@H]1c2cccnc2",
    "Lactic acid": "CC(O)C(=O)O",
    "Benzene": "c1ccccc1",
}

def calculate_properties(name, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES. Please check the molecular notation."
    formula = rdMolDescriptors.CalcMolFormula(mol)
    row = {
        "Molecule": name,
        "SMILES": Chem.MolToSmiles(mol),
        "Formula": formula,
        "Molecular weight (g/mol)": round(Descriptors.MolWt(mol), 3),
        "Exact mass (Da)": round(Descriptors.ExactMolWt(mol), 4),
        "Heavy atoms": int(mol.GetNumHeavyAtoms()),
        "Total atoms": int(Chem.AddHs(mol).GetNumAtoms()),
        "H-bond donors": int(Lipinski.NumHDonors(mol)),
        "H-bond acceptors": int(Lipinski.NumHAcceptors(mol)),
        "LogP": round(Crippen.MolLogP(mol), 3),
        "TPSA (Å²)": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "Rotatable bonds": int(Lipinski.NumRotatableBonds(mol)),
        "Ring count": int(rdMolDescriptors.CalcNumRings(mol)),
    }
    return (mol, row), None

def load_library():
    rows, mols = [], {}
    for name, smiles in MOLECULES.items():
        result, error = calculate_properties(name, smiles)
        if result:
            mol, row = result
            rows.append(row)
            mols[name] = (mol, smiles)
    return pd.DataFrame(rows), mols

def rule_of_five(row):
    return {
        "Molecular weight ≤ 500": row["Molecular weight (g/mol)"] <= 500,
        "LogP ≤ 5": row["LogP"] <= 5,
        "H-bond donors ≤ 5": row["H-bond donors"] <= 5,
        "H-bond acceptors ≤ 10": row["H-bond acceptors"] <= 10,
    }

with st.sidebar:
    st.markdown("## 🧪 Molecular Studio")
    st.caption("CHEMINFORMATICS WORKSPACE")
    page = st.radio("WORKSPACE", [
        "Dashboard", "Molecule Explorer", "Compare Molecules",
        "SMILES Analyzer", "Dataset & Export", "Project Report"
    ])
    st.markdown("---")
    st.caption("Python · Streamlit · RDKit · Plotly")
    st.caption("Educational project; computed descriptors are not experimental measurements.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">Computational chemistry · Advanced mini-project</div>
  <h1>🧪 Molecular Property Studio</h1>
  <p>Explore molecular structures, calculate physicochemical descriptors, compare compounds, and export results in one interactive workspace.</p>
</div>
""", unsafe_allow_html=True)

df, molecules = load_library()
if df.empty:
    st.error("No molecules loaded. Check your RDKit installation.")
    st.stop()

if "custom_name" not in st.session_state:
    st.session_state.custom_name = "My molecule"
if "custom_smiles" not in st.session_state:
    st.session_state.custom_smiles = "CCO"

if page == "Dashboard":
    st.markdown('<div class="section-title">Molecular analytics dashboard</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Compounds in library", len(df))
    m2.metric("Descriptor types", 8)
    largest = df.loc[df["Molecular weight (g/mol)"].idxmax()]
    m3.metric("Largest molecule", largest["Molecule"], f'{largest["Molecular weight (g/mol)"]:.2f} g/mol')
    m4.metric("Average LogP", f'{df["LogP"].mean():.2f}')

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Molecular weight comparison")
        fig = px.bar(df.sort_values("Molecular weight (g/mol)", ascending=False),
                     x="Molecule", y="Molecular weight (g/mol)",
                     color="Molecular weight (g/mol)", color_continuous_scale="Teal")
        fig.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white",
                          font_color="#17263d", margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Lipophilicity vs polar surface")
        fig = px.scatter(df, x="LogP", y="TPSA (Å²)", hover_name="Molecule",
                         size="Molecular weight (g/mol)", color="Molecule")
        fig.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white",
                          font_color="#17263d", showlegend=False,
                          margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Compound library</div>', unsafe_allow_html=True)
    st.dataframe(df[["Molecule", "Formula", "Molecular weight (g/mol)", "LogP",
                     "TPSA (Å²)", "H-bond donors", "H-bond acceptors", "Rotatable bonds"]],
                 use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Rule-of-Five descriptor screen</div>', unsafe_allow_html=True)
    screen = []
    for _, row in df.iterrows():
        checks = rule_of_five(row)
        item = {"Molecule": row["Molecule"], "Checks passed": f'{sum(checks.values())}/4'}
        item.update({k: "Pass" if v else "Review" for k, v in checks.items()})
        screen.append(item)
    st.dataframe(pd.DataFrame(screen), use_container_width=True, hide_index=True)
    st.caption("A rough educational descriptor screen; it does not establish drug safety or efficacy.")

elif page == "Molecule Explorer":
    st.markdown('<div class="section-title">Molecule explorer</div>', unsafe_allow_html=True)
    selected = st.selectbox("Choose a compound", list(molecules.keys()))
    mol, smiles = molecules[selected]
    row = df.loc[df["Molecule"] == selected].iloc[0]
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.subheader(selected)
        st.write(f"**Molecular formula:** {row['Formula']}")
        st.caption("SMILES notation")
        st.code(smiles, language=None)
        st.image(Draw.MolToImage(mol, size=(500, 380)),
                 caption=f"2D structure of {selected}", use_container_width=True)

    with right:
        st.subheader("Calculated descriptors")
        a, b = st.columns(2)
        a.metric("Molecular weight", f'{row["Molecular weight (g/mol)"]:.3f} g/mol')
        b.metric("Exact mass", f'{row["Exact mass (Da)"]:.4f} Da')
        a, b = st.columns(2)
        a.metric("LogP", f'{row["LogP"]:.3f}')
        b.metric("TPSA", f'{row["TPSA (Å²)"]:.2f} Å²')
        a, b, c = st.columns(3)
        a.metric("H-bond donors", int(row["H-bond donors"]))
        b.metric("H-bond acceptors", int(row["H-bond acceptors"]))
        c.metric("Rotatable bonds", int(row["Rotatable bonds"]))

        st.markdown("#### Structure summary")
        st.write(f'Heavy atoms: **{int(row["Heavy atoms"])}** · Total atoms: **{int(row["Total atoms"])}** · Rings: **{int(row["Ring count"])}**')

        st.markdown("#### Rule-of-Five descriptor check")
        checks = rule_of_five(row)
        passed = sum(checks.values())
        st.progress(passed / len(checks))
        st.write(f"**{passed}/4 descriptor cutoffs passed**")
        for label, result in checks.items():
            st.write(("✅ " if result else "⚠️ ") + label)
        st.caption("This heuristic is for learning only; it does not show that a molecule is an effective drug.")

    st.info("LogP estimates lipophilicity; TPSA summarizes polar surface area. These descriptors support comparison but do not prove biological activity.")

elif page == "Compare Molecules":
    st.markdown('<div class="section-title">Compare molecular properties</div>', unsafe_allow_html=True)
    chosen = st.multiselect("Select 2–5 compounds", list(molecules.keys()),
                            default=["Caffeine", "Aspirin", "Ethanol"], max_selections=5)
    if len(chosen) < 2:
        st.warning("Select at least two compounds.")
    else:
        comp = df[df["Molecule"].isin(chosen)].copy()
        c1, c2, c3 = st.columns(3)
        high = comp.loc[comp["Molecular weight (g/mol)"].idxmax()]
        low = comp.loc[comp["Molecular weight (g/mol)"].idxmin()]
        c1.metric("Highest molecular weight", high["Molecule"], f'{high["Molecular weight (g/mol)"]:.2f} g/mol')
        c2.metric("Lowest molecular weight", low["Molecule"], f'{low["Molecular weight (g/mol)"]:.2f} g/mol')
        c3.metric("Mean LogP", f'{comp["LogP"].mean():.2f}')

        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Molecular weight")
            fig = px.bar(comp, x="Molecule", y="Molecular weight (g/mol)", color="Molecule")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#17263d")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.subheader("LogP vs TPSA")
            fig = px.scatter(comp, x="LogP", y="TPSA (Å²)", text="Molecule",
                             size="Molecular weight (g/mol)", color="Molecule")
            fig.update_traces(textposition="top center")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              font_color="#17263d", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.download_button("Download comparison CSV", comp.to_csv(index=False).encode("utf-8"),
                           "molecule_comparison.csv", "text/csv")

elif page == "SMILES Analyzer":
    st.markdown('<div class="section-title">Analyze your own molecule</div>', unsafe_allow_html=True)
    st.write("Enter a name and a SMILES string. Example: CCO represents ethanol.")
    with st.form("custom_molecule_form"):
        name = st.text_input("Molecule name", value=st.session_state.custom_name)
        smiles = st.text_input("SMILES notation", value=st.session_state.custom_smiles)
        submit = st.form_submit_button("Analyze molecule", use_container_width=True)

    if submit:
        result, error = calculate_properties(name.strip() or "Custom molecule", smiles.strip())
        if error:
            st.error(error)
        else:
            st.session_state.custom_name = name.strip() or "Custom molecule"
            st.session_state.custom_smiles = smiles.strip()
            mol, row = result
            left, right = st.columns([1, 1.25], gap="large")
            with left:
                st.subheader(row["Molecule"])
                st.write(f'**Formula:** {row["Formula"]}')
                st.code(row["SMILES"], language=None)
                st.image(Draw.MolToImage(mol, size=(500, 380)),
                         caption="Generated 2D molecular structure", use_container_width=True)
            with right:
                st.subheader("Calculated descriptors")
                st.metric("Molecular weight", f'{row["Molecular weight (g/mol)"]:.3f} g/mol')
                st.metric("LogP", f'{row["LogP"]:.3f}')
                st.metric("TPSA", f'{row["TPSA (Å²)"]:.3f} Å²')
                st.dataframe(pd.DataFrame([{"Descriptor": k, "Value": v}
                                           for k, v in row.items()
                                           if k not in ["Molecule", "SMILES", "Formula"]],
                                          ), use_container_width=True, hide_index=True)
                st.download_button("Download molecule CSV",
                                   pd.DataFrame([row]).to_csv(index=False).encode("utf-8"),
                                   "custom_molecule_analysis.csv", "text/csv")

elif page == "Dataset & Export":
    st.markdown('<div class="section-title">Dataset and export center</div>', unsafe_allow_html=True)
    query = st.text_input("Search by molecule name, formula, or SMILES")
    filtered = df.copy()
    if query.strip():
        q = query.strip()
        mask = (
            filtered["Molecule"].str.contains(q, case=False, na=False)
            | filtered["Formula"].str.contains(q, case=False, na=False)
            | filtered["SMILES"].str.contains(q, case=False, na=False)
        )
        filtered = filtered[mask]
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Download dataset CSV", filtered.to_csv(index=False).encode("utf-8"),
                       "molecular_property_studio_results.csv", "text/csv")

    st.markdown("### Descriptor guide")
    st.markdown("""
- **Molecular weight:** sum of atomic weights.
- **Exact mass:** mass based on the most abundant isotopes.
- **LogP:** estimated octanol/water partition coefficient.
- **TPSA:** topological polar surface area.
- **H-bond donors/acceptors:** structural features relevant to hydrogen bonding.
- **Rotatable bonds:** approximate number of rotatable bonds.
""")

elif page == "Project Report":
    st.markdown('<div class="section-title">Project report and viva notes</div>', unsafe_allow_html=True)
    st.subheader("Abstract")
    st.write("Molecular Property Studio is an interactive cheminformatics application built with Python, Streamlit, and RDKit. It analyzes molecular structures represented by SMILES and calculates molecular weight, exact mass, LogP, topological polar surface area, hydrogen-bond donor and acceptor counts, rotatable bonds, and ring count. It displays 2D structures, supports compound comparisons, visualizes descriptor relationships, and exports results to CSV.")
    st.subheader("Objective")
    st.write("To develop an interactive platform that represents chemical structures, calculates molecular descriptors, and compares physicochemical properties through visual analysis.")
    st.subheader("Methodology")
    st.write("1. Load a predefined or user-provided SMILES string. 2. Parse it using RDKit. 3. Calculate molecular descriptors. 4. Draw a 2D structure. 5. Compare molecules using charts and tables. 6. Export results to CSV.")
    st.subheader("Observations")
    st.write("Descriptors vary with molecular structure and functional groups. Molecules with similar molecular weights can differ in LogP and TPSA. Descriptors support preliminary comparison but do not independently predict safety, efficacy, or experimental behavior.")
    st.subheader("Conclusion")
    st.write("The project combines cheminformatics, molecular visualization, interactive analysis, and data export in a compact scientific tool. Future work could add 3D conformer generation and molecular mechanics.")
    st.subheader("Limitations")
    st.write("The application does not run quantum-chemistry calculations, docking, experimental validation, or drug-efficacy prediction.")
    st.subheader("Viva questions")
    st.markdown("""
**What is SMILES?** A text notation representing molecular structure.

**What is LogP?** An estimate of partitioning between octanol and water.

**What is TPSA?** A descriptor of topological polar surface area.

**Why RDKit?** It provides molecule parsing, depiction, and descriptor calculations.

**Does this prove a molecule is an effective drug?** No. Descriptors alone are not evidence of efficacy.
""")

st.markdown("---")
st.caption("Molecular Property Studio · RDKit-powered educational cheminformatics project")
