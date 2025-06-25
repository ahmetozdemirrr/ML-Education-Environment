// frontend/src/services/datasetService.js

const API_BASE_URL = 'http://localhost:8000/api';

export class DatasetService {
  static async getAvailableDatasets() {
    try {
      const response = await fetch(`${API_BASE_URL}/datasets`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      return data.datasets;
    } catch (error) {
      console.error('Error fetching available datasets:', error);
      throw error;
    }
  }

  static async getDatasetVisualization(datasetName, method = 'pca') {
    try {
      const response = await fetch(`${API_BASE_URL}/datasets/${datasetName}/visualize?method=${method}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      return data;
    } catch (error) {
      console.error('Error fetching dataset visualization:', error);
      throw error;
    }
  }

  static async getDatasetInfo(datasetName) {
    try {
      const response = await fetch(`${API_BASE_URL}/datasets/${datasetName}/info`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      return data;
    } catch (error) {
      console.error('Error fetching dataset info:', error);
      throw error;
    }
  }

  static getDatasetNameFromPath(datasetPath) {
    // Convert dataset display names to filenames
    const mapping = {
      'Iris Dataset': 'iris.csv',
      'Wine Dataset': 'wine_data.csv',
      'Breast Cancer Wisconsin': 'breast_cancer_wisconsin.csv',
      'Two Moons Dataset': 'two_moons_data.csv',
      'Digits Dataset': 'digits_data.csv',
      'Pima Indians Diabetes': 'pima_indians_diabetes.csv',
      'Synthetic Hard Classification': 'synthetic_classification_hard.csv',
      'Banknote Authentication': 'banknote_authentication.csv',
      'Haberman Survival': 'haberman_survival.csv',
      'Mushroom Data': 'mushroom_data.csv',
      'Abalone Classification': 'abalone_classification.csv',
      'Bike Sharing Hourly': 'bike_sharing_hourly_classification.csv',
      'MNIST Subset': 'mnist_subset_7k.csv'
    };

    // If it's already a filename, return as is
    if (datasetPath.endsWith('.csv')) {
      return datasetPath;
    }

    // Try to find mapping
    return mapping[datasetPath] || `${datasetPath.toLowerCase().replace(/\s+/g, '_')}.csv`;
  }

  static async validateDatasetExists(datasetName) {
    try {
      const datasets = await this.getAvailableDatasets();
      return datasets.some(d => d.filename === datasetName || d.name === datasetName);
    } catch (error) {
      console.error('Error validating dataset:', error);
      return false;
    }
  }
}

export default DatasetService;
