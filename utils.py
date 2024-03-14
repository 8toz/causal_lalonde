import os
import pandas as pd
import networkx as nx
import json

TREATMENT_PATH = "./data/ab_test"
OBSERVATIONAL_PATH = "./data/observational_data"
LALONDE_COLUMNS = ["treatment",  "age", "education", "black" , "hispanic", "married", "nodegree", "re74" , "re78"]
NSWRE_COLUMNS = ["treatment",  "age", "education", "black" , "hispanic", "married", "nodegree", "re74", "re75", "re78"]
INTEGER_COLS =  ["treatment", "age", "education", "black", "hispanic", "married", "nodegree"]

CAUSAL_DISCOVERY_PATH = "./causal_discovery_graphs"


def load_data() -> pd.DataFrame:
    """
    Extracts Lalonde original experiment. 
    Extracts the subset of Lalonde used by NSWRE (experimental)
    Extracts the observational control groups of CPS and PSID
    """
    lalonde_control = pd.read_csv(os.path.join(TREATMENT_PATH, "lalonde_control.txt"), sep='\s+',index_col=None, header=None, names=LALONDE_COLUMNS)
    lalonde_treated = pd.read_csv(os.path.join(TREATMENT_PATH, "lalonde_treated.txt"), sep='\s+',index_col=None, header=None, names=LALONDE_COLUMNS)
    lalonde_original = pd.concat([lalonde_control, lalonde_treated], axis=0)
    lalonde_original[INTEGER_COLS] = lalonde_original[INTEGER_COLS].astype("int64")
    assert(lalonde_original.shape[0] == 722)

    nswre_control = pd.read_csv(os.path.join(TREATMENT_PATH, "nswre74_control.txt"), sep='\s+',index_col=None, header=None, names=NSWRE_COLUMNS)
    nswre_treated = pd.read_csv(os.path.join(TREATMENT_PATH, "nswre74_treated.txt"), sep='\s+',index_col=None, header=None, names=NSWRE_COLUMNS)
    nswre_df = pd.concat([nswre_control, nswre_treated], axis=0)
    nswre_df[INTEGER_COLS] = nswre_df[INTEGER_COLS].astype("int64")
    assert(nswre_df.shape[0] == 445)

    cps_control = pd.read_csv(os.path.join(OBSERVATIONAL_PATH, "cps_controls.txt"), sep='\s+',index_col=None, header=None, names=NSWRE_COLUMNS)
    cps_control[INTEGER_COLS] = cps_control[INTEGER_COLS].astype("int64")
    assert(cps_control.shape[0] == 15992)
    psid_control = pd.read_csv(os.path.join(OBSERVATIONAL_PATH, "psid_controls.txt"), sep='\s+',index_col=None, header=None, names=NSWRE_COLUMNS)
    psid_control[INTEGER_COLS] = psid_control[INTEGER_COLS].astype("int64")
    assert(psid_control.shape[0] == 2490)

    cps2_control = pd.read_csv(os.path.join(OBSERVATIONAL_PATH, "cps2_controls.txt"), sep='\s+',index_col=None, header=None, names=NSWRE_COLUMNS)
    cps2_control[INTEGER_COLS] = cps2_control[INTEGER_COLS].astype("int64")
    assert(cps_control.shape[0] == 15992)
    psid2_control = pd.read_csv(os.path.join(OBSERVATIONAL_PATH, "psid2_controls.txt"), sep='\s+',index_col=None, header=None, names=NSWRE_COLUMNS)
    psid2_control[INTEGER_COLS] = psid2_control[INTEGER_COLS].astype("int64")
    assert(psid_control.shape[0] == 2490)

    cps3_control = pd.read_csv(os.path.join(OBSERVATIONAL_PATH, "cps3_controls.txt"), sep='\s+',index_col=None, header=None, names=NSWRE_COLUMNS)
    cps3_control[INTEGER_COLS] = cps3_control[INTEGER_COLS].astype("int64")
    assert(cps_control.shape[0] == 15992)
    psid3_control = pd.read_csv(os.path.join(OBSERVATIONAL_PATH, "psid3_controls.txt"), sep='\s+',index_col=None, header=None, names=NSWRE_COLUMNS)
    psid3_control[INTEGER_COLS] = psid3_control[INTEGER_COLS].astype("int64")
    assert(psid_control.shape[0] == 2490)

    return lalonde_original, nswre_df, cps_control, psid_control, cps2_control, cps3_control, psid2_control, psid3_control

def extract_CD_graph(name : str) -> nx.Graph:
    """
    Extracts a graph from a json file based on a match with the filename
    """
    G = nx.Graph()
    for file in os.listdir(CAUSAL_DISCOVERY_PATH):
        if file.endswith(".json") and str(name) in file:
            # Load graph data from JSON file
            with open(os.path.join(CAUSAL_DISCOVERY_PATH, file), "r") as f:
                data = json.load(f)
                # Create a temporary graph from the loaded data
                G = nx.node_link_graph(data)
        
    return G

def draw_graph(G : nx.Graph) -> None:
    """
    Plots a graph by using a spring layout algorithm to facilitate visualization
    """
    pos = nx.spring_layout(G, k=100, seed=42) # Seed to make it deterministic
    colors = ['red'  if (y,x) in G.edges() else 'black' for (x,y) in G.edges()]
    nx.draw(G, with_labels=True, node_size=3500, node_color='w', edgecolors ='black', pos=pos, edge_color=colors)
    return None


