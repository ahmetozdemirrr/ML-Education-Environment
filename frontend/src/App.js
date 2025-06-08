/* ./frontend/src/App.js */

import React, { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import './App.css';
import GlobalSettingsPanel from './GlobalSettingsPanel';
import isEqual from 'lodash/isEqual';


/* Machine Learning Models: */
const initialModelsData = [
  {
    id: 'dt_clf',
    name: 'Decision Tree',
    task_type: 'classification',
    checked: false,
    description: 'A Decision Tree is a supervised learning model that creates decision rules by recursively splitting data based on feature values. It resembles a flowchart where each internal node represents a test on an attribute, each branch represents an outcome of the test, and each leaf node represents a class label. They are intuitive, easy to interpret ("white-box" model), and can handle both numerical and categorical data.',
    params: [
      { name: 'Criterion', type: 'select', options: ['Gini', 'Entropy', 'Log Loss'], defaultValue: 'Gini' },
      { name: 'Max Depth', type: 'number', defaultValue: '', placeholder: 'None (integer)', min: 1, step: 1 },
      { name: 'Min Samples Split', type: 'number', defaultValue: 2, min: 2, step: 1 },
      { name: 'Min Samples Leaf', type: 'number', defaultValue: 1, min: 1, step: 1 },
    ]
  },
  {
    id: 'lr',
    name: 'Logistic Regression',
    task_type: 'classification',
    checked: false,
    description: 'Logistic Regression is a statistical model used to predict the probability of a binary outcome (e.g., yes/no, 0/1). Though it has "regression" in its name, it\'s a classification algorithm. It uses a logistic (sigmoid) function to map the output of a linear equation to a value between 0 and 1, representing the probability. It can also be extended for multi-class classification (Multinomial Logistic Regression).',
    params: [
      { name: 'Penalty', type: 'select', options: ['L2', 'L1', 'ElasticNet', 'None'], defaultValue: 'L2' },
      { name: 'C (Reg. Strength)', type: 'number', defaultValue: 1.0, min: 0.01, step: 0.01, format: 'float' },
      { name: 'Solver', type: 'select', options: ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'], defaultValue: 'lbfgs' },
      { name: 'Max Iterations', type: 'number', defaultValue: 100, min: 10, step: 10 },
    ]
  },
  {
    id: 'svm_clf',
    name: 'SVM',
    task_type: 'classification',
    checked: false,
    description: 'Support Vector Machine (SVM) is a powerful classification algorithm that finds the optimal hyperplane that best separates data points of different classes in a high-dimensional space. The "optimal" hyperplane is the one that maximizes the margin (the distance between the hyperplane and the closest data points from each class, called support vectors). SVMs are effective in high-dimensional spaces and can model non-linear relationships using the "kernel trick" (e.g., RBF, polynomial kernels).',
    params: [
      { name: 'C (Reg. Param)', type: 'number', defaultValue: 1.0, min: 0.01, step: 0.01, format: 'float' },
      { name: 'Kernel', type: 'select', options: ['RBF', 'Linear', 'Poly', 'Sigmoid'], defaultValue: 'RBF' },
      { name: 'Gamma', type: 'select', options: ['Scale', 'Auto'], defaultValue: 'Scale' },
      { name: 'Degree (Poly Kernel)', type: 'number', defaultValue: 3, min: 1, step: 1, condition: { param: 'Kernel', value: 'Poly' } },
    ]
  },
  {
    id: 'knn_clf',
    name: 'K-Nearest Neighbor',
    task_type: 'classification',
    checked: false,
    description: 'K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm (or lazy learning algorithm) where a new data point is classified based on the majority class of its K closest neighbors in the feature space. It does not learn an explicit discriminative function from the training data but relies on feature similarity. It\'s non-parametric, meaning it makes no assumptions about the underlying data distribution. Performance can be sensitive to the choice of K, the distance metric, and feature scaling.',
    params: [
      { name: 'N Neighbors', type: 'slider', defaultValue: 5, min: 1, max: 50, step: 1 },
      { name: 'Weights', type: 'select', options: ['Uniform', 'Distance'], defaultValue: 'Uniform' },
      { name: 'Algorithm', type: 'select', options: ['auto', 'ball_tree', 'kd_tree', 'brute'], defaultValue: 'auto' },
      { name: 'Metric', type: 'select', options: ['minkowski', 'euclidean', 'manhattan', 'chebyshev'], defaultValue: 'minkowski' },
    ]
  },
  {
    id: 'ann_clf',
    name: 'Artificial Neural Network',
    task_type: 'classification',
    checked: false,
    description: 'An Artificial Neural Network (ANN) is a computational model inspired by the structure and functioning of biological neural networks in the human brain. It consists of interconnected nodes (neurons) organized in layers: an input layer, one or more hidden layers, and an output layer. ANNs learn by adjusting the synaptic weights (strengths of connections) between neurons during a training process. They are highly effective for modeling complex, non-linear relationships and are the foundation of deep learning. However, they can be computationally intensive and are often considered "black-box" models due to their complex internal workings.',
    params: [
      { name: 'Hidden Layer Sizes', type: 'text', defaultValue: '100', placeholder: 'e.g., 100 or 50,20' },
      { name: 'Activation', type: 'select', options: ['ReLU', 'Tanh', 'Logistic', 'Identity'], defaultValue: 'ReLU' },
      { name: 'Solver', type: 'select', options: ['Adam', 'SGD', 'L-BFGS'], defaultValue: 'Adam' },
      { name: 'Alpha (L2 Penalty)', type: 'number', defaultValue: 0.0001, min: 0, step: 0.0001, format: 'float' },
      { name: 'Learning Rate (SGD)', type: 'select', options: ['Constant', 'InvScaling', 'Adaptive'], defaultValue: 'Constant', condition: { param: 'Solver', value: 'SGD' } },
    ]
  },
];

/* Datasets for train models: */
const availableDatasets = [
  {
    id: 'iris',
    name: 'Iris Dataset',
    description: 'A classic and widely-used dataset in machine learning and statistics. It contains 150 samples of Iris flowers, each with 4 features: sepal length, sepal width, petal length, and petal width (all in cm). The task is to predict one of three species: Iris Setosa, Iris Versicolour, or Iris Virginica. It\'s often used for practicing classification algorithms.'
  },
  {
    id: 'two_moons_data',
    name: 'Two Moons (Synthetic)',
    description: 'A synthetic, 2-dimensional dataset often used for visualizing and testing classification algorithms. It consists of 400 data points forming two interleaving half-circle shapes (moons), making it a simple benchmark for models that can learn non-linear decision boundaries. Each moon represents a different class.'
  },
  {
    id: 'wine_data',
    name: 'Wine Dataset',
    description: 'This dataset contains the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars (grape varieties). It includes 178 samples, each with 13 chemical constituents (features) like alcohol content, malic acid, ash, etc. The goal is to classify the wines into one of the three cultivars. It can highlight the importance of feature scaling due to varying ranges of feature values.'
  },
  {
    id: 'breast_cancer_wisconsin',
    name: 'Breast Cancer Wisconsin',
    description: 'A well-known dataset for binary classification, used to predict whether a breast mass is benign or malignant. It contains 569 samples. Features (30 in total) are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, describing characteristics of the cell nuclei present (e.g., radius, texture, perimeter, area). Classes are typically encoded as malignant and benign.'
  },
  {
    id: 'digits_data',
    name: 'Digits Dataset',
    description: 'A small, popular dataset included with Scikit-learn for practicing handwritten digit recognition. It consists of 1797 samples, where each sample is an 8x8 pixel grayscale image of a handwritten digit (0 through 9). This results in 64 features per sample, representing the pixel intensities. It serves as a simpler alternative to the larger MNIST dataset.'
  },
  {
    id: 'haberman_survival',
    name: 'Haberman\'s Survival Data',
    description: 'This dataset contains cases from a study conducted on the survival of patients who had undergone surgery for breast cancer. It includes 306 samples and 3 numerical features: age of patient at time of operation, patient\'s year of operation (year - 1900), and number of positive axillary nodes detected. The binary classification task is to predict survival status (survived 5 years or longer, or died within 5 years).'
  },
  {
    id: 'pima_indians_diabetes',
    name: 'Pima Indians Diabetes',
    description: 'This dataset is used to predict whether or not a patient (female, at least 21 years old, of Pima Indian heritage) has diabetes based on 8 diagnostic medical predictor variables. These features include number of pregnancies, glucose concentration, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age. It contains 768 samples and 2 classes (0 for no diabetes, 1 for diabetes).'
  },
  {
    id: 'banknote_authentication',
    name: 'Banknote Authentication',
    description: 'This dataset is used for distinguishing between genuine and forged banknote-like specimens. Data were extracted from 400x400 pixel images of genuine and forged banknotes. For each image, 4 features were extracted using a Wavelet Transform tool: variance of Wavelet Transformed image, skewness, kurtosis, and entropy of the image. It contains 1372 samples with 2 classes (genuine or forged).'
  },
  {
    id: 'mushroom_data',
    name: 'Mushroom Dataset',
    description: 'This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family. Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended (this latter class was combined with poisonous). It contains 8124 samples and 22 categorical features (e.g., cap-shape, cap-surface, odor), which are numerically encoded. WARNING: This dataset is purely for pattern recognition practice and should NOT be used to determine the edibility of real mushrooms.'
  },
  {
    id: 'synthetic_classification_hard',
    name: 'Synthetic Classification',
    description: 'An artificially generated dataset specifically designed to be moderately challenging for classification algorithms. It comprises 1000 samples, 3 classes, and 20 features, which include a mix of informative features, redundant features (linear combinations of informative ones), and noisy features (unrelated to the class). This makes it useful for evaluating model robustness and feature selection techniques.'
  },
  {
    id: 'abalone_classification',
    name: 'Abalone Age (Classification)',
    description: 'This dataset is used to predict the age of abalone (a type of sea snail) from physical measurements. The age is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope – a tedious task. The original "rings" feature (which directly indicates age) has been binned into 3 age categories (e.g., young, adult, old) for this classification task. It has 4177 samples and 8 features (one of which, "Sex", is categorical and typically encoded). '
  },
  {
    id: 'bike_sharing_hourly_classification',
    name: 'Bike Sharing Hourly Demand',
    description: 'This dataset contains hourly bike rental counts from a bike-sharing system in Washington D.C. over two years. For this classification version, the original continuous count of bike rentals has been binned into 3 demand levels (e.g., low, medium, high). It includes approximately 17,379 samples and 12 features, such as season, month, hour, holiday, weekday, weather conditions (temperature, humidity, windspeed), providing a rich set for predicting rental demand categories.'
  },
  {
    id: 'mnist_subset_7k',
    name: 'MNIST Digits (Subset 7k)',
    description: 'A 7,000-sample stratified subset of MNIST for handwritten digit recognition. 784 features (28x28 images), 10 classes. Good for quicker experiments.'
  },
  {
    id: 'mnist_full_70k',
    name: 'MNIST Digits (Full 70k)',
    description: 'The complete MNIST dataset for handwritten digit recognition. Contains 70,000 samples (60k training, 10k testing in original split). 784 features (28x28 images), 10 classes. Ideal for testing on larger data but takes longer to process.'
  },
];

const initialStaticDatasets = availableDatasets.map(d => ({ ...d, checked: false }));

const allMetrics = 
{
  classification: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
};

function ParameterPopup({ model, onClose, onSaveRequest, existingConfigs = [], editingConfig = null })
{
  const [paramsState, setParamsState] = useState({});
  const [selectedMetricsState, setSelectedMetricsState] = useState([]);
  const availableMetrics = model ? allMetrics[model.task_type] || [] : [];

  useEffect(() => 
  {
    if (model) 
    {
      const initialParams = {};

      if (editingConfig && editingConfig.params) {
        model.params.forEach(param => {
          initialParams[param.name] = editingConfig.params[param.name] !== undefined ? editingConfig.params[param.name] : param.defaultValue;
        });
        setSelectedMetricsState(editingConfig.metrics || (availableMetrics.length > 0 ? [availableMetrics[0]] : []));
      } 

      else {
        model.params.forEach(param => {
          initialParams[param.name] = param.defaultValue;
        });
        setSelectedMetricsState(availableMetrics.length > 0 ? [availableMetrics[0]] : []);
      }
      setParamsState(initialParams);
    }
  }, 
  [model, editingConfig]);

  const handleInputChange = (paramName, value) => 
  { 
    setParamsState(prevParams => ({ ...prevParams, [paramName]: value })); 
  };

  const handleMetricChangePopup = (event) => 
  {
    const { value, checked } = event.target;

    setSelectedMetricsState(prevMetrics => 
    {
      if (checked) { return [...prevMetrics, value]; }
      else { return prevMetrics.filter(metric => metric !== value); }
    });
  };

  const handleSaveAttempt = () => 
  {
    const currentData = { params: paramsState, metrics: selectedMetricsState };
    onSaveRequest(model.id, currentData, editingConfig ? editingConfig.id : null);
  };

  if (!model) return null;

  const shouldDisplay = (param) => 
  {
    if (!param.condition) return true;
    const conditionParamValue = paramsState[param.condition.param];
    return conditionParamValue === param.condition.value;
  };

  return (
    <div className="popup-overlay">
      <div className="popup-content">
        <button className="popup-close-button" onClick={onClose}>&times;</button>
        <h3>{editingConfig ? 'Edit' : model.name + ' - Configure'} Instance</h3>
        <h4>Hyperparameters</h4>
        <div className="param-list">
          {model.params.filter(shouldDisplay).map((param) => (
            <div key={param.name} className="param-item">
              <label htmlFor={param.name}>{param.name}:</label>
              {param.type === 'select' && ( <select id={param.name} value={paramsState[param.name] ?? param.defaultValue} onChange={(e) => handleInputChange(param.name, e.target.value)} > {param.options.map(option => (<option key={option} value={option}>{option}</option>))} </select> )}
              {param.type === 'number' && ( <input type="number" id={param.name} value={paramsState[param.name] ?? ''} placeholder={param.placeholder || ''} min={param.min} max={param.max} step={param.step || (param.format === 'float' ? 0.0001 : 1)} onChange={(e) => handleInputChange(param.name, e.target.value === '' ? '' : (param.format === 'float' ? parseFloat(e.target.value) : parseInt(e.target.value, 10)))} /> )}
              {param.type === 'slider' && ( <div className="slider-container"> <input type="range" id={param.name} min={param.min} max={param.max} step={param.step} value={paramsState[param.name] ?? param.defaultValue} onChange={(e) => handleInputChange(param.name, parseInt(e.target.value, 10))} /> <span className="slider-value">{paramsState[param.name] ?? param.defaultValue}</span> </div> )}
              {param.type === 'text' && ( <input type="text" id={param.name} value={paramsState[param.name] ?? param.defaultValue} placeholder={param.placeholder || ''} onChange={(e) => handleInputChange(param.name, e.target.value)} /> )}
            </div>
          ))}
        </div>
        {availableMetrics.length > 0 && (
          <div className="metrics-section-popup">
            <h4>Evaluation Metrics</h4>
            <div className="metrics-checkboxes-popup">
              {availableMetrics.map(metric => ( <label key={metric} className="metric-label-popup"> <input type="checkbox" value={metric} checked={selectedMetricsState.includes(metric)} onChange={handleMetricChangePopup} /> {metric} </label> ))}
            </div>
          </div>
        )}
        <button className="popup-ok-button" onClick={handleSaveAttempt}>{editingConfig ? 'Update' : 'Save'} Configuration</button>
      </div>
    </div>
  );
}

function ConfirmationPopup({ modelName, onConfirm, onCancel }) 
{
  return ( <div className="popup-overlay confirmation-overlay"> <div className="popup-content confirmation-content"> <h4>Add Another Instance?</h4> <p>Do you want to add another instance of <strong>{modelName}</strong> with different settings?</p> <div className="confirmation-buttons"> <button onClick={onConfirm} className="confirm-yes-button">Yes, Add Another</button> <button onClick={onCancel} className="confirm-no-button">No, Finish</button> </div> </div> </div> );
}

function App() 
{
  const [models] = useState(initialModelsData);
  const [datasets, setDatasets] = useState(initialStaticDatasets);
  const [tooltipContent, setTooltipContent] = useState('');
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0, opacity: 0 });
  const tooltipTimeoutRef = useRef(null);

  const [selectedModelForPopup, setSelectedModelForPopup] = useState(null);
  const [editingConfigState, setEditingConfigState] = useState(null);

  const [savedParams, setSavedParams] = useState({});
  const [globalSettings, setGlobalSettings] = useState({
    useCrossValidation: false,
    cvFolds: 5,
    useTrainTestSplit: true,
    testSplitRatio: 0.2,
    randomSeedType: "fixed",
    applyFeatureScaling: true,
    scalerType: "standard"
  });
  const [isLoading, setIsLoading] = useState(false);
  const [confirmationState, setConfirmationState] = useState({ show: false, model: null });
  const simulationPanelRef = useRef(null);
  const API_BASE_URL = "http://localhost:8000";

  const handleMouseEnter = (event, descriptionText) => 
  {
    if (tooltipTimeoutRef.current) { clearTimeout(tooltipTimeoutRef.current); tooltipTimeoutRef.current = null; }
    const targetRect = event.currentTarget.getBoundingClientRect();
    const tooltipNominalHeight = 50; 
    const tooltipWidth = 260; 
    const gap = 15; 
    const viewportPadding = 10; 
    let top = targetRect.top + window.scrollY - tooltipNominalHeight - gap; 
    let left = targetRect.left + window.scrollX + (targetRect.width / 2);
    let transform = 'translateX(-50%)'; 
    if (top < window.scrollY + viewportPadding) { top = targetRect.bottom + window.scrollY + gap; }
    if (left - (tooltipWidth / 2) < viewportPadding) { left = viewportPadding + (tooltipWidth / 2) ; }
    else if (left + (tooltipWidth / 2) > window.innerWidth - viewportPadding) { left = window.innerWidth - viewportPadding - (tooltipWidth / 2); }
    setTooltipPosition({ top: top, left: left, transform: transform, opacity: 1 });
    setTooltipContent(descriptionText);
  };

  const handleMouseLeave = () => 
  {
    tooltipTimeoutRef.current = setTimeout(() => 
    {
        setTooltipPosition(prev => ({ ...prev, opacity: 0 }));
        setTimeout(() => { if (!tooltipTimeoutRef.current) return; setTooltipContent(''); tooltipTimeoutRef.current = null; }, 200); 
    }, 100); 
  };

  const configsAreEqual = (config1, config2) => isEqual(config1.params, config2.params) && isEqual(config1.metrics, config2.metrics);
  
  const handleModelCheckboxChange = (modelId) => 
  {
    const modelToConfigure = models.find(m => m.id === modelId);

    if (selectedModelForPopup?.id === modelId && !editingConfigState) {
        setSelectedModelForPopup(null);
        setEditingConfigState(null);
    } 

    else if (modelToConfigure) {
        setSelectedModelForPopup(modelToConfigure);
        setEditingConfigState(null);
    }
  };

  const handleDatasetCheckboxChange = (datasetId) => setDatasets(datasets.map(d => d.id === datasetId ? { ...d, checked: !d.checked } : d ));
  
  const closePopup = () => 
  {
    setSelectedModelForPopup(null);
    setEditingConfigState(null);
  };

  const handleSaveConfiguration = (modelId, newData, configToEditId = null) => 
  {
    const existingConfigsForModel = savedParams[modelId] || [];
    
    if (configToEditId) {
        const updatedConfigs = existingConfigsForModel.map(config =>
            config.id === configToEditId ? { ...config, ...newData } : config
        );
        setSavedParams(prevParams => ({ ...prevParams, [modelId]: updatedConfigs }));
        console.log(`Configuration ${configToEditId} updated for ${modelId}:`, newData);
    } 

    else {
        const isDuplicate = existingConfigsForModel.some(existingConfig => configsAreEqual(existingConfig, newData));
        
        if (isDuplicate) {
            alert("These settings are identical to a previously configured instance for this model.");
            return false;
        }
        const newConfigId = `${modelId}_config_${Date.now()}`;
        const newConfig = { id: newConfigId, ...newData };
        setSavedParams(prevParams => ({ ...prevParams, [modelId]: [...existingConfigsForModel, newConfig] }));
        console.log(`Configuration ${newConfigId} saved for ${modelId}:`, newConfig);
    }
    return true;
  };

  const handleSaveAndConfirm = (modelId, newData, configToEditId = null) => 
  {
    const success = handleSaveConfiguration(modelId, newData, configToEditId);
    
    if (success) 
    {
        if (configToEditId) {
            closePopup();
        } 

        else {
            const model = models.find(m => m.id === modelId);
            setConfirmationState({ show: true, model: model });
        }
    }
  };

  const handleConfirmYes = () => setConfirmationState({ show: false, model: null });
  const handleConfirmNo = () => { setConfirmationState({ show: false, model: null }); closePopup(); };

  const handleDeleteConfiguration = (modelId, configIdToDelete) => 
  {
    setSavedParams(prevParams => {
        const updatedConfigsForModel = (prevParams[modelId] || []).filter(config => config.id !== configIdToDelete);
        if (updatedConfigsForModel.length === 0) {
            const { [modelId]: _, ...restParams } = prevParams;
            return restParams;
        }
        return { ...prevParams, [modelId]: updatedConfigsForModel };
    });
    console.log(`Configuration ${configIdToDelete} deleted for model ${modelId}.`);
  };

  const handleEditConfiguration = (modelId, configIdToEdit) => 
  {
      const modelToConfigure = models.find(m => m.id === modelId);
      const configToEdit = (savedParams[modelId] || []).find(c => c.id === configIdToEdit);
      
      if (modelToConfigure && configToEdit) {
          setSelectedModelForPopup(modelToConfigure);
          setEditingConfigState(configToEdit);
      }
  };

  const handleResetWorkspace = () =>
  {
    if (window.confirm("Are you sure you want to reset all selections and configurations?")) {
        setDatasets(initialStaticDatasets.map(d => ({ ...d, checked: false })));
        setSavedParams({});
        setGlobalSettings({
          useCrossValidation: false,
          cvFolds: 5,
          useTrainTestSplit: true,
          testSplitRatio: 0.2,
          randomSeedType: "fixed",
          applyFeatureScaling: true,
          scalerType: "standard"
        });
        setSelectedModelForPopup(null);
        setEditingConfigState(null);
        setConfirmationState({ show: false, model: null });
        console.log("Workspace reset.");
    }
  };

  const handleGlobalSettingsChange = (settingName, value) => {
    setGlobalSettings(prevSettings => {
      const newSettings = { ...prevSettings, [settingName]: value };

      // Mutual exclusive logic: sadece biri aktif olabilir
      if (settingName === 'useCrossValidation' && value) {
        newSettings.useTrainTestSplit = false;
      }
      if (settingName === 'useTrainTestSplit' && value) {
        newSettings.useCrossValidation = false;
      }

      // En az birinin aktif olması gerekiyor
      if (!newSettings.useCrossValidation && !newSettings.useTrainTestSplit) {
        newSettings.useTrainTestSplit = true;
      }

      console.log("Global Settings Updated in App: ", newSettings);
      return newSettings;
    });
  };

  // Backend'e İstek Gönderme Fonksiyonu
  const sendSimulationRequest = async (endpoint, actionType) => 
  {
    console.log(`--- ${actionType} Button Clicked ---`);
    setIsLoading(true);

    const allConfiguredModels = Object.values(savedParams).flat(); // Tüm konfigürasyonları tek bir diziye al
    const selectedDatasets = datasets.filter(d => d.checked);

    if (allConfiguredModels.length === 0 || selectedDatasets.length === 0) {
      alert(`Please configure at least one model instance and select at least one dataset to ${actionType.toLowerCase()}.`);
      setIsLoading(false); 
      return;
    }
    const requests = [];

    // Her bir konfigürasyon ve her bir dataset için istek oluştur
    for (const config of allConfiguredModels) 
    {
        const originalModel = models.find(m => config.id.startsWith(m.id)); // Orijinal model bilgisini bul
        if (!originalModel) continue;

        for (const dataset of selectedDatasets) 
        {
          const randomSeed = globalSettings.randomSeedType === "fixed" ? 42 : null;
          const payload = 
          {
              algorithm: originalModel.name,
              params: {
                  ...config.params,
                  selectedMetrics: config.metrics,
                  frontend_config_id: config.id
              },
              dataset: dataset.id,
              global_settings: {
                ...globalSettings,
                randomSeed: randomSeed // randomSeedType'a göre dinamik olarak ayarlanıyor
              }
          };
          console.log(`Sending ${actionType} request for ${originalModel.name} (Config: ${config.id}) on ${dataset.name}`);
          console.log("Payload:", JSON.stringify(payload, null, 2));

          const requestPromise = fetch(`${API_BASE_URL}${endpoint}`, {
              method: 'POST', headers: { 'Content-Type': 'application/json', }, body: JSON.stringify(payload),
          })
          .then(async response => {
              if (!response.ok) {
                  let errorData = { detail: `HTTP error! Status: ${response.status}` };
                  try { errorData = await response.json(); } catch (e) { console.error("Could not parse error response", e); }
                  console.error(`Error ${actionType}ing ${originalModel.name} (Config: ${config.id}) on ${dataset.name}: ${response.status} ${response.statusText}`, errorData);
                  throw new Error(errorData.detail || `HTTP error ${response.status}`);
              } return response.json();
          })
          .then(data => {
              console.log(`${actionType} Response for ${originalModel.name} (Config: ${config.id}) on ${dataset.name}:`, data);
              // Sonuçları işlemek için config.id'yi de ekleyebiliriz
              return { status: 'fulfilled', configId: config.id, model: originalModel.name, dataset: dataset.name, data: data };
          })
          .catch(error => {
              console.error(`Workspace error during ${actionType} for ${originalModel.name} (Config: ${config.id}) on ${dataset.name}:`, error);
                  return { status: 'rejected', configId: config.id, model: originalModel.name, dataset: dataset.name, reason: error.message };
          });
          requests.push(requestPromise);
        }
    }

    try 
    {
        const results = await Promise.allSettled(requests);
        console.log(`${actionType} - All requests settled:`, results);
        setIsLoading(false);
        const failures = results.filter(result => result.status === 'rejected');
        
        if (failures.length > 0) {
            console.warn(`${failures.length} request(s) failed during ${actionType}.`);
            alert(`${failures.length} simulation task(s) failed. Check console for details.`);
        } 

        else {
            alert(`${actionType} requests completed successfully for all configurations! Check backend console.`);
        }
    } 

    catch (error) 
    {
        console.error(`Error managing ${actionType} requests:`, error);
        setIsLoading(false);
        alert(`An error occurred while sending ${actionType} requests.`);
    }
  };

  const handleTrain = () => { sendSimulationRequest('/train', 'Train'); };
  const handleEvaluate = () => { sendSimulationRequest('/train', 'Evaluate'); };

  const toggleFullScreen = async () => 
  {
    const elem = simulationPanelRef.current;
    if (!elem) return;

    if (!document.fullscreenElement) 
    {
      try 
      {
        if (elem.requestFullscreen) { await elem.requestFullscreen(); }
        else if (elem.webkitRequestFullscreen) { await elem.webkitRequestFullscreen(); }
        else if (elem.msRequestFullscreen) { await elem.msRequestFullscreen(); }
        else { alert('Fullscreen API is not supported by your browser.'); }
      } 
      catch (err) { console.error(`Error attempting to enable full-screen mode: ${err.message} (${err.name})`); }
    } 

    else 
    {
      try 
      {
        if (document.exitFullscreen) { await document.exitFullscreen(); }
        else if (document.webkitExitFullscreen) { await document.webkitExitFullscreen(); }
        else if (document.msExitFullscreen) { await document.msExitFullscreen(); }
        else { alert('Fullscreen API is not supported by your browser.'); }
      } 
      catch (err) { console.error(`Error attempting to exit full-screen mode: ${err.message} (${err.name})`); }
    }
  };

  return (
    <div className="app-container">
      <div className="left-column">
        <div className="panel models-panel">
          <h2>Available Models</h2>
          <ul>
            {models.map((model) => {
              const configs = savedParams[model.id] || [];
              const isConfigured = configs.length > 0;
              return (
                <li key={model.id} className="model-item">
                  <label className="model-label-container" title={isConfigured ? `${configs.length} configuration(s) set. Click to add/edit.` : 'Select to configure model'}>
                    <input type="checkbox" checked={isConfigured} onChange={() => handleModelCheckboxChange(model.id)} />
                    <span className="model-name">{model.name} {isConfigured ? `(${configs.length})` : ''}</span>
                  </label>
                  {model.description && (
                    <div className="info-icon-container" onMouseEnter={(e) => handleMouseEnter(e, model.description)} onMouseLeave={handleMouseLeave}>
                      <span className="info-icon" aria-label="Model Information">ⓘ</span>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
        <GlobalSettingsPanel settings={globalSettings} onChange={handleGlobalSettingsChange} />
      </div>

      <div className="panel simulation-panel" ref={simulationPanelRef}>
        <div className="simulation-panel-header">
          <h2>Simulation Screen</h2>
          <button onClick={toggleFullScreen} className="fullscreen-button" title="Toggle Fullscreen">
             <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"> <path d="M1.5 1a.5.5 0 0 0-.5.5v4a.5.5 0 0 1-1 0v-4A1.5 1.5 0 0 1 1.5 0h4a.5.5 0 0 1 0 1h-4zM10 .5a.5.5 0 0 1 .5-.5h4A1.5 1.5 0 0 1 16 1.5v4a.5.5 0 0 1-1 0v-4a.5.5 0 0 0-.5-.5h-4a.5.5 0 0 1-.5-.5zM.5 10a.5.5 0 0 1 .5.5v4a.5.5 0 0 0 .5.5h4a.5.5 0 0 1 0 1h-4A1.5 1.5 0 0 1 0 14.5v-4a.5.5 0 0 1 .5-.5zm15 0a.5.5 0 0 1 .5.5v4a1.5 1.5 0 0 1-1.5 1.5h-4a.5.5 0 0 1 0-1h4a.5.5 0 0 0 .5-.5v-4a.5.5 0 0 1 .5-.5z"/> </svg>
          </button>
        </div>
        <div className="simulation-panel-content">
          {isLoading && <div className="loading-indicator">Running Simulations...</div>}
          <p>( Currently Empty - Simulation results will appear here )</p>
          <div className="simulation-summary">
            <div>
              <h4>Configured Model Instances:</h4>
              {Object.keys(savedParams).length > 0 ? (
                <ul className="configured-instances-list">
                  {Object.entries(savedParams).map(([modelId, configs]) =>
                    configs.map((config, index) => {
                      const originalModel = models.find(m => m.id === modelId);
                      return (
                        <li key={config.id} className="config-item">
                          <div className="config-item-name">
                            <span><strong>{originalModel?.name || modelId}</strong> (Instance {index + 1})</span>
                          </div>
                          <div className="config-item-actions">
                            <button 
                              onClick={() => handleEditConfiguration(modelId, config.id)} 
                              className="config-action-button edit-button" 
                              title="Edit Configuration">
                              ✏️ {/* Kalem ikonu */}
                            </button>
                            <button 
                              onClick={() => handleDeleteConfiguration(modelId, config.id)} 
                              className="config-action-button delete-button" 
                              title="Delete Configuration">
                              🗑️ {/* Çöp kutusu ikonu */}
                            </button>
                          </div>
                          <details className="config-details">
                            <summary>Details</summary>
                            <pre>
                              ID: {config.id}{"\n"}
                              Params: {JSON.stringify(config.params, null, 1)}{"\n"}
                              Metrics: {JSON.stringify(config.metrics, null, 1)}
                            </pre>
                          </details>
                        </li>
                      );
                    })
                  )}
                </ul>
              ) : ( <small>No model instances configured yet.</small> )}
            </div>
            <div><h4>Selected Datasets:</h4><ul>{datasets.filter(d => d.checked).map(d => <li key={d.id}>{d.name}</li>)}</ul>{datasets.filter(d => d.checked).length === 0 && <small>No datasets selected.</small>}</div>
            <div><h4>Global Settings:</h4><pre>{JSON.stringify(globalSettings, null, 2)}</pre></div>
          </div>
        </div>
      </div>

      <div className="right-column">
        <div className="panel datasets-panel">
          <h2>Datasets</h2>
          {datasets.length > 0 ? (
            <ul>
              {datasets.map((dataset) => (
                <li key={dataset.id} className="dataset-item">
                  <label className="dataset-label-container">
                    <input type="checkbox" checked={dataset.checked} onChange={() => handleDatasetCheckboxChange(dataset.id)} />
                    <span className="dataset-name">{dataset.name}</span>
                  </label>
                  {dataset.description && (
                    <div className="info-icon-container" onMouseEnter={(e) => handleMouseEnter(e, dataset.description)} onMouseLeave={handleMouseLeave}>
                      <span className="info-icon" aria-label="Dataset Information">ⓘ</span>
                    </div>
                  )}
                </li>
              ))}
            </ul>
          ) : ( <div style={{padding: '10px', textAlign:'center'}}>No datasets defined.</div> )}
        </div>
        <div className="action-buttons-bottom">
          <button className="train-button" onClick={handleTrain} disabled={isLoading}>{isLoading ? 'Processing...' : 'Train'}</button>
          <button className="evaluate-button" onClick={handleEvaluate} disabled={isLoading}>{isLoading ? 'Processing...' : 'Evaluate'}</button>
          {/* YENİ: Reset Butonu */}
          <button className="reset-button" onClick={handleResetWorkspace} disabled={isLoading}>Reset Workspace</button>
        </div>
      </div>

      {selectedModelForPopup && ( 
        <ParameterPopup 
            model={selectedModelForPopup} 
            onClose={closePopup} 
            onSaveRequest={handleSaveAndConfirm} 
            existingConfigs={savedParams[selectedModelForPopup.id] || []}
            editingConfig={editingConfigState} // Düzenlenecek konfigürasyonu prop olarak geçir
        /> 
      )}
      {confirmationState.show && confirmationState.model && ( <ConfirmationPopup modelName={confirmationState.model.name} onConfirm={handleConfirmYes} onCancel={handleConfirmNo} /> )}

      {tooltipContent && createPortal(
        <div
          className="tooltip"
          style={{
            position: 'fixed',
            top: `${tooltipPosition.top}px`,
            left: `${tooltipPosition.left}px`,
            transform: tooltipPosition.transform || 'none',
            opacity: tooltipPosition.opacity,
            transition: 'opacity 0.2s ease-in-out',
            pointerEvents: 'none',
            zIndex: 1100,
          }}
        >
          {tooltipContent}
        </div>,
        document.body
      )}
    </div>
  );
}

export default App;
