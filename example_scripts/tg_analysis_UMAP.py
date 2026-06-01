# Script to analyze Tg using UMAP and clustering on MD trajectory data
# Baseline imports
import statistics as stats
import numpy as np
import pandas as pd
import statistics as stats
import glob
import os
# MDAnalysis for trajectory handling
import MDAnalysis as mda
# Distance calculation function from MDAnalysis
from MDAnalysis.lib.distances import self_distance_array
#Standard scaling and clustering funcitons from Sci-kit learn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
# UMAP for dimensionality reduction - from umap-learn package
import umap
# Optional - for progress bars
from tqdm import tqdm
# For silhouette score 
from sklearn.metrics import silhouette_score

directory= os.getcwd()
results_dir = f'{directory}/{results}'
os.chdir(results_dir)
construct = '<CONSTRUCT NAME HERE>'
# Initialize an empty DataFrame to store all observations
df_full_results = pd.DataFrame()

# Detect all DCD files
dcd_files = glob.glob("trajectory_NPT_dcd_tempramp*.dcd")
dcd_files.sort() # Sort by temperature order - important for later analysis

# Load the universe for the first DCD file to get atom info
u = mda.Universe("em.gro", dcd_files[0], guess_bonds=True) 
# guess_bonds=True to ensure bonds are assigned if .gro files used
# This may not be necessary if using other coordinate files e.g. .pdb files with CONECT records

# Create directory for analysis outputs - not necessary but makes plotting easier
dir_name = f'../UMAP_outputs/{file}/'
os.makedirs(dir_name, exist_ok=True)

# Loop through each chain in the universe
for chain in u.atoms.fragments[:25]: # Select just main ensemble not residual molecules
    chain_idx = chain.indices  # Get indices of atoms in the chain
    f_index = chain.atoms[0].fragindex
    chain_df = pd.DataFrame()  # DataFrame for the current chain
    # Loop through each DCD file
    for temp_file in tqdm(dcd_files, desc=f"Processing DCD files for chain {f_index}"):
        u = mda.Universe("em.gro", temp_file)  # Load the universe for the current DCD file
        # Extract temperature from the file name
        temp = float(temp_file.split('_')[-1].split('.dcd')[0].replace('-', '.').replace('K', ''))
        # Loop through each frame in the trajectory
        for ts in u.trajectory:
            # Select the chain atoms
            chain_temp = u.atoms[chain_idx].select_atoms('not type H') # Select only heavy atoms
            pos = chain_temp.positions  # Get the positions of the atoms in the chain

            # Compute pairwise distances
            dist_matrix = self_distance_array(pos, box=u.dimensions)

            # Create a temporary DataFrame for the current frame
            df_temp = pd.DataFrame(dist_matrix).T
            df_temp['Temp'] = temp  # Add temperature as a column
            df_temp.set_index('Temp', inplace=True)

            # Append to the full DataFrame
            chain_df = pd.concat([chain_df, df_temp], ignore_index=True)

    # Apply standard scaler to standardise the data
    # Standard Scaling gives each feature a mean of 0 and a standard deviation of 1
    stand_scaler = StandardScaler()
    df_standardised = chain_df.copy()
    df_standardised = stand_scaler.fit_transform(df_standardised)

    # UMAP dimensionality reduction
    standard_embedding = umap.UMAP(random_state=42).fit_transform(df_standardised)

    # Create a DataFrame of UMAP components
    umap_df = pd.DataFrame(
        data=standard_embedding, 
        columns=['UMAP1', 'UMAP2']
    )
    # Agglomerative Clustering using Sckit-learn - can adjust n_clusters as needed for dataset size
    silhouette_dict = {}
    rows = []
    for c_num in range(2, 11):
        umap2_df_iter = umap_df.copy()

        agg = AgglomerativeClustering(n_clusters=c_num)
        labels = agg.fit_predict(umap2_df_iter)

        assert len(labels) == len(umap2_df_iter)

        umap2_df_iter[f'Cluster_{c_num}'] = labels
        grid_size = 10  # granularity of the grid
        umap2_df_iter['temp'] = np.arange(200,500.5,0.5)
        #umap_df.set_index('temp', inplace=True)
        import numpy as np
        # Create a new column for grid squares
        umap2_df_iter['Grid_X'] = (umap2_df_iter['UMAP1'] // grid_size).astype(int)
        umap2_df_iter['Grid_Y'] = (umap2_df_iter['UMAP2'] // grid_size).astype(int)
        sil = silhouette_score(umap_df, labels, metric="euclidean")
 
        grid_counts = (
            umap2_df_iter.groupby([f'Cluster_{c_num}', 'Grid_X', 'Grid_Y'])
            .size()
            .reset_index(name='Count')
        )

        most_dense_cluster = grid_counts.loc[
            grid_counts['Count'].idxmax(), f'Cluster_{c_num}'
        ]

        umap2_df_iter[f'State_{c_num}'] = umap2_df_iter[f'Cluster_{c_num}'].apply(
            lambda x: 'glassy' if x == most_dense_cluster else 'rubbery'
        )

        umap2_df_iter['Cluster_Change'] = (
            umap2_df_iter[f'State_{c_num}'] != umap2_df_iter[f'State_{c_num}'].shift()
        )
        umap2_df_iter.loc[0, 'Cluster_Change'] = False

        change_points = umap2_df_iter[umap2_df_iter['Cluster_Change']]['temp']

        tg_value = None
        for change_point_index in change_points.index:
            next_states = umap2_df_iter[f'State_{c_num}'].iloc[
                change_point_index:change_point_index + 10
            ]
            if all(next_states == 'rubbery'):
                tg_value = umap2_df_iter.loc[change_point_index, 'temp']
                break

        rows.append({
            "chain": f_index,
            "k": c_num,
            "Tg": tg_value,
            "silhouette": sil
        })
        silhouette_dict[c_num]=sil
    best_c_num, max_sil = max(silhouette_dict.items(), key=lambda kv: kv[1])
    agg_clustering = AgglomerativeClustering(n_clusters=best_c_num)       
    clusters = agg_clustering.fit_predict(umap_df)
    umap_df['Cluster'] = clusters
    umap_df['temp'] = np.arange(200,500.5,0.5) # Assuming temperature ramp from 200K to 500K with 0.5K increments
    umap_df.set_index('temp', inplace=True)

    # Determining the most dense cluster using a grid-based approach
    grid_size = 10  # granularity of the grid
    # Create a new column for grid squares
    umap_df['Grid_X'] = (umap_df['UMAP1'] // grid_size).astype(int)
    umap_df['Grid_Y'] = (umap_df['UMAP2'] // grid_size).astype(int)
    # Group by cluster and grid square, and count the number of points in each square
    grid_counts = umap_df.groupby(['Cluster', 'Grid_X', 'Grid_Y']).size().reset_index(name='Count')
    # Find the most dense cluster (the cluster with the highest count in any square)
    most_dense_cluster = grid_counts.loc[grid_counts['Count'].idxmax(), 'Cluster']
    # Label the data points
    umap_df['State'] = umap_df['Cluster'].apply(
        lambda x: 'glassy' if x == most_dense_cluster else 'rubbery')

    umap_df.to_csv(f"{dir_name}clustering_results_chain_{f_index}.csv")
    umap_df.reset_index(inplace=True)


    # Find where the state changes
    umap_df['State_Change'] = umap_df['State'] != umap_df['State'].shift()
    umap_df.loc[0, 'State_Change'] = False  # Ensure the first data point is not marked as a change

    # Extract the temperatures where the cluster changes
    change_points = umap_df[umap_df['State_Change']]['temp']

    # Assume the first change point is Tg - this may need refinement based on data
    tg = change_points.iloc[0]
    # Store the results in temporary DataFrame
    df_chain = pd.DataFrame([f_index, tg], index=['chain', 'Tg']).T
    # Append to full results DataFrame
    df_full_results = pd.concat([df_full_results, df_chain])
    df_full_results.to_csv(f'{dir_name}all_chains.csv')

# Calculate and append mean Tg for entire melt
df_full_results.loc[len(df_chain)] = ['mean', stats.mean(df_full_results['Tg'])]
df_full_results.to_csv(f'{dir_name}all_chains.csv')
print(f"mean Tg = {stats.mean(df_full_results['Tg'])}")

