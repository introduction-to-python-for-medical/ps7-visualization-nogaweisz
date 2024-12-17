import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    """Load the Spotify dataset"""
    df = pd.read_csv('spotify-2023.csv', encoding='latin-1')
    # Convert streams to numeric, removing any commas
    df['streams'] = pd.to_numeric(df['streams'].str.replace(',', ''), errors='coerce')
    return df

def plot_histograms(df):
    """Create histograms for key features"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distribution of Key Features', fontsize=16)
    
    # Streams histogram
    axes[0, 0].hist(df['streams'], bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Streams')
    axes[0, 0].set_xlabel('Streams')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Release year histogram
    axes[0, 1].hist(df['released_year'], bins=20, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Distribution of Release Years')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Artist count histogram
    axes[1, 0].hist(df['artist_count'], bins=15, color='salmon', edgecolor='black')
    axes[1, 0].set_title('Distribution of Artist Count')
    axes[1, 0].set_xlabel('Number of Artists')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Danceability histogram
    axes[1, 1].hist(df['danceability_%'], bins=30, color='lightgray', edgecolor='black')
    axes[1, 1].set_title('Distribution of Danceability')
    axes[1, 1].set_xlabel('Danceability %')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_scatter_plots(df):
    """Create scatter plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Feature Relationships', fontsize=16)
    
    # Streams vs Spotify Playlists
    axes[0, 0].scatter(df['in_spotify_playlists'], df['streams'], alpha=0.5, color='blue')
    axes[0, 0].set_title('Streams vs Spotify Playlists')
    axes[0, 0].set_xlabel('Number of Spotify Playlists')
    axes[0, 0].set_ylabel('Streams')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Streams vs Apple Playlists
    axes[0, 1].scatter(df['in_apple_playlists'], df['streams'], alpha=0.5, color='red')
    axes[0, 1].set_title('Streams vs Apple Playlists')
    axes[0, 1].set_xlabel('Number of Apple Playlists')
    axes[0, 1].set_ylabel('Streams')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Danceability vs Energy
    axes[1, 0].scatter(df['danceability_%'], df['energy_%'], alpha=0.5, color='green')
    axes[1, 0].set_title('Danceability vs Energy')
    axes[1, 0].set_xlabel('Danceability %')
    axes[1, 0].set_ylabel('Energy %')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Valence vs Acousticness
    axes[1, 1].scatter(df['valence_%'], df['acousticness_%'], alpha=0.5, color='purple')
    axes[1, 1].set_title('Valence vs Acousticness')
    axes[1, 1].set_xlabel('Valence %')
    axes[1, 1].set_ylabel('Acousticness %')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_correlation_matrix(df):
    """Create correlation matrix"""
    key_features = [
        'streams', 
        'in_spotify_playlists', 
        'in_apple_playlists',
        'danceability_%',
        'valence_%',
        'energy_%',
        'acousticness_%'
    ]
    
    corr_matrix = df[key_features].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(im)
    
    ax.set_xticks(np.arange(len(key_features)))
    ax.set_yticks(np.arange(len(key_features)))
    ax.set_xticklabels(key_features, rotation=45, ha='right')
    ax.set_yticklabels(key_features)
    
    for i in range(len(key_features)):
        for j in range(len(key_features)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                         ha='center', va='center', color='black')
    
    plt.title('Correlation Matrix of Key Features')
    plt.tight_layout()
    
    return fig, corr_matrix

def main():
    try:
        # Load data
        df = load_data()
        print("Data loaded successfully!")
        print(df.head())
        # Print the total number of rows in the dataset
        print(f"\nTotal number of songs in the dataset: {len(df)}")
        
        # Create and save histograms
        hist_fig = plot_histograms(df)
        hist_fig.savefig('histograms.png')
        plt.close(hist_fig)
        
        # Create and save scatter plots
        scatter_fig = plot_scatter_plots(df)
        scatter_fig.savefig('scatter_plots.png')
        plt.close(scatter_fig)
        
        # Create and save correlation matrix
        corr_fig, corr_matrix = create_correlation_matrix(df)
        corr_fig.savefig('correlation_matrix.png')
        plt.close(corr_fig)
        
        # Print correlation insights
        print("\nKey Correlation Insights:")
        print("\nStrong Positive Correlations (> 0.5):")
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i,j] > 0.5:
                    print(f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: {corr_matrix.iloc[i,j]:.2f}")
        
        print("\nStrong Negative Correlations (< -0.5):")
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i,j] < -0.5:
                    print(f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: {corr_matrix.iloc[i,j]:.2f}")
        
        print("\nAnalysis complete! All visualizations have been saved as PNG files.")
        
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main(
