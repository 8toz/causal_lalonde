import os
import pandas as pd
import networkx as nx
import json
import dowhy
import random
import numpy as np
import re

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

    nswre_control = pd.read_csv(os.path.join(TREATMENT_PATH, "nswre74_control.txt"), sep='\s+',index_col=None, header=None, names=NSWRE_COLUMNS)
    nswre_treated = pd.read_csv(os.path.join(TREATMENT_PATH, "nswre74_treated.txt"), sep='\s+',index_col=None, header=None, names=NSWRE_COLUMNS)
    nswre_df = pd.concat([nswre_control, nswre_treated], axis=0)
    nswre_df[INTEGER_COLS] = nswre_df[INTEGER_COLS].astype("int64")
  
    cps_control = pd.read_csv(os.path.join(OBSERVATIONAL_PATH, "cps_controls.txt"), sep='\s+',index_col=None, header=None, names=NSWRE_COLUMNS)
    cps_control[INTEGER_COLS] = cps_control[INTEGER_COLS].astype("int64")
   
    psid_control = pd.read_csv(os.path.join(OBSERVATIONAL_PATH, "psid_controls.txt"), sep='\s+',index_col=None, header=None, names=NSWRE_COLUMNS)
    psid_control[INTEGER_COLS] = psid_control[INTEGER_COLS].astype("int64")


    cps2_control = pd.read_csv(os.path.join(OBSERVATIONAL_PATH, "cps2_controls.txt"), sep='\s+',index_col=None, header=None, names=NSWRE_COLUMNS)
    cps2_control[INTEGER_COLS] = cps2_control[INTEGER_COLS].astype("int64")

    psid2_control = pd.read_csv(os.path.join(OBSERVATIONAL_PATH, "psid2_controls.txt"), sep='\s+',index_col=None, header=None, names=NSWRE_COLUMNS)
    psid2_control[INTEGER_COLS] = psid2_control[INTEGER_COLS].astype("int64")

    cps3_control = pd.read_csv(os.path.join(OBSERVATIONAL_PATH, "cps3_controls.txt"), sep='\s+',index_col=None, header=None, names=NSWRE_COLUMNS)
    cps3_control[INTEGER_COLS] = cps3_control[INTEGER_COLS].astype("int64")

    psid3_control = pd.read_csv(os.path.join(OBSERVATIONAL_PATH, "psid3_controls.txt"), sep='\s+',index_col=None, header=None, names=NSWRE_COLUMNS)
    psid3_control[INTEGER_COLS] = psid3_control[INTEGER_COLS].astype("int64")

    return lalonde_original, nswre_df, cps_control, psid_control, cps2_control, cps3_control, psid2_control, psid3_control


def load_combinations(df: pd.DataFrame, cps_control: pd.DataFrame, cps2_control: pd.DataFrame, cps3_control: pd.DataFrame,  psid_control: pd.DataFrame, psid2_control: pd.DataFrame, psid3_control: pd.DataFrame) -> pd.DataFrame:
    """
    We create all possible combinations of datasets with the treatment subset we have (NSWRE T=1).
    """
    treated = df[df["treatment"]==1]
    nswre_cps = pd.concat([treated, cps_control], axis=0)
    nswre_cps2 = pd.concat([treated, cps2_control], axis=0)
    nswre_cps3 = pd.concat([treated, cps3_control], axis=0)
    nswre_psid = pd.concat([treated, psid_control], axis=0)
    nswre_psid2 = pd.concat([treated, psid2_control], axis=0)
    nswre_psid3 = pd.concat([treated, psid3_control], axis=0)

    return nswre_cps, nswre_cps2, nswre_cps3, nswre_psid, nswre_psid2, nswre_psid3



def extract_CD_graph(name: str) -> nx.Graph:
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

def draw_graph(G: nx.Graph) -> None:
    """
    Plots a graph by using a spring layout algorithm to facilitate visualization
    """
    pos = nx.spring_layout(G, k=100, seed=42) # Seed to make it deterministic
    colors = ['red'  if (y,x) in G.edges() else 'black' for (x,y) in G.edges()]
    nx.draw(G, with_labels=True, node_size=3500, font_size=10, node_color='w', edgecolors ='black', pos=pos, edge_color=colors)
    return None

def graph_to_json(graph: nx.Graph, filename: str):
    """ 
    Saves a Graph in JSON format into its corresponding path
    Parameters:
    Graph 
    File Name (without .json extension) 
    """
    data = nx.node_link_data(graph)
    file_path = os.path.join(CAUSAL_DISCOVERY_PATH, filename + ".json")
    with open(file_path, "w") as f:
        json.dump(data, f)
                
    return "Graph saved at "+ file_path

def probabilistic_causal_effect(combinations_dict: dict, graph: nx.Graph, treatment="treatment", outcome="re78") -> pd.DataFrame:
    """
    Creates a Probabilistic Causal Model and Outputs the different treatment effects given a graph and a DataFrame
    Input - Dict of DataFrames
    Returns - Dict of Average Treatment Effect
    """
    random.seed(42)
    np.random.seed(42)

    causal_effects = {}
    for df in combinations_dict:
        lalonde_graph_model = dowhy.gcm.ProbabilisticCausalModel(graph)
        dowhy.gcm.auto.assign_causal_mechanisms(lalonde_graph_model, combinations_dict[df])
        dowhy.gcm.fit(lalonde_graph_model, combinations_dict[df])

        effect = dowhy.gcm.average_causal_effect(lalonde_graph_model, outcome, 
                            interventions_alternative={treatment: lambda x: 1},
                            interventions_reference={treatment: lambda x: 0},
                            num_samples_to_draw=1000
                            )
        causal_effects[df] = effect

    df = pd.DataFrame(list(causal_effects.items()), columns=['composite_df', 'probabilistic_effect'])
    df.set_index('composite_df', inplace=True)

    return df

    
def get_all_directed_paths(graph: nx.Graph) -> list:
    """
    Function to get all directed paths in a directed NetworkX graph.
    """
    all_paths = []
    for start_node in graph.nodes():
        for end_node in graph.nodes():
            if start_node != end_node:
                paths = list(nx.all_simple_paths(graph, source=start_node, target=end_node))
                all_paths.extend(paths)
    cleaned = []
    for item in all_paths:
        if (item[0], item[-1]) not in cleaned:
            cleaned.append((item[0], item[-1]))

    return cleaned

def indentify_effects(graph: nx.Graph, df: pd.DataFrame, pairs=None) -> pd.DataFrame:
    """
    Identify the effect given a graph. 
    If pairs is None it will iterate over all directed combinations of nodes 
    identifying its effect. 
    """
    if pairs is None:
        pairs = get_all_directed_paths(graph)
   
    dfs = []
    for pair in pairs:
        model = dowhy.CausalModel(data=df, graph=graph, treatment=pair[0], outcome=pair[1])
        identified_effect = model.identify_effect(proceed_when_unidentifiable=True)
        backdoor = np.NaN if identified_effect.estimands["backdoor"] is None else identified_effect.get_backdoor_variables()
        frontdoor = np.NaN if identified_effect.estimands["frontdoor"] is None else identified_effect.get_frontdoor_variables()
        instrumental_variables = np.NaN if identified_effect.estimands["iv"] is None else identified_effect.get_instrumental_variables()

        effects = pd.DataFrame({
            'treatment': [pair[0]],
            'outcome': [pair[1]],
            'backdoor': [backdoor],
            'frontdoor': [frontdoor],
            'instrumental_variables': [instrumental_variables]
        })
        dfs.append(effects)

    auxiliar_df = pd.concat(dfs, ignore_index=True)

    backdoor_group = auxiliar_df[["treatment", "outcome", "backdoor"]][~auxiliar_df["backdoor"].isna()]
    frontdoor_group = auxiliar_df[["treatment", "outcome","frontdoor"]][~auxiliar_df["frontdoor"].isna()]
    iv_group =  auxiliar_df[["treatment", "outcome","instrumental_variables"]][~auxiliar_df["instrumental_variables"].isna()]

    # backdoor_group = backdoor_group.dropna(axis=1)
    # frontdoor_group = frontdoor_group.dropna(axis=1)
    # iv_group = iv_group.dropna(axis=1)

    return backdoor_group, frontdoor_group, iv_group


def causal_effect(combinations_dict: dict, graph: nx.Graph, methods: list, refuter_list: list, treatment="treatment", outcome="re78") -> pd.DataFrame:
    result_dict = {}
    for df_key, df_value in combinations_dict.items():
        #print(df_key)
        auxiliar_df = {}
        model = dowhy.CausalModel(data=df_value, treatment=treatment, graph=graph, outcome=outcome)
        identified_effect = model.identify_effect(proceed_when_unidentifiable=True)

        auxiliar_df["treatment"] = treatment
        auxiliar_df["outcome"] = outcome
        for method in methods:
            #print(method)
            lalonde_estimate = model.estimate_effect(identified_effect, 
                                                    method_name=method,
                                                    # target_units="ate",
                                                    # method_params={"weighting_scheme":"ips_weight"}
                                                    )
            auxiliar_df[method] = lalonde_estimate.value

            for refute in refuter_list:
                #print("Refuting...")
                refute_result =  model.refute_estimate(identified_effect, lalonde_estimate, method_name=refute)
                match = re.search(r'New effect:([-+]?\d*\.?\d+)', str(refute_result))
                pvalue = re.search(r'p value:([-+]?\d*\.?\d+)', str(refute_result))
                new_effect_value = str(np.round(float(match.group(1)),2)) +" (" + str(np.round(float(pvalue.group(1)),2))+")"
                auxiliar_df[refute+"_"+method.split("_")[-1]] = new_effect_value
       
        result_dict[df_key] = auxiliar_df

    result_df = pd.DataFrame.from_dict(result_dict, orient='index')

    #prob_estimation = probabilistic_causal_effect(combinations_dict, graph)
    #result_df = result_df.join(prob_estimation)

    return result_df

def baseline_1999(combinations_dict: dict, methods: list, refuter_list: list, treatment="treatment", outcome="re78"):
    result_dict = {}
    for df_key, df_value in combinations_dict.items():
        auxiliar_df = {}
        model = dowhy.CausalModel(data=df_value, treatment=treatment, outcome=outcome)
        identified_effect = model.identify_effect(proceed_when_unidentifiable=True)

        auxiliar_df["treatment"] = treatment
        auxiliar_df["outcome"] = outcome
        for method in methods:
            lalonde_estimate = model.estimate_effect(identified_effect, 
                                                    method_name=method,
                                                    # target_units="ate",
                                                    # method_params={"weighting_scheme":"ips_weight"}
                                                    )
            auxiliar_df[method] = lalonde_estimate.value

            for refute in refuter_list:
                refute_result =  model.refute_estimate(identified_effect, lalonde_estimate, method_name=refute)
                match = re.search(r'New effect:([-+]?\d*\.?\d+)', str(refute_result))
                new_effect_value = float(match.group(1))
                auxiliar_df[refute+method] = new_effect_value
       
        result_dict[df_key] = auxiliar_df

    result_df = pd.DataFrame.from_dict(result_dict, orient='index')

    return result_df