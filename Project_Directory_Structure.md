# Demand Forecasting Project Structure map

This is a comprehensive map of the project directory. It shows how every folder and file fits together to build the complete AI forecasting system.

```text
Demand_Forecast/                 # Root Directory
│
├── README.md                    # Project overview and setup instructions.
├── Comprehensive_File_Explanations.md # Detailed professor-level explanations of each file.
├── gen_alpha_fig.py             # Generates the 2-panel alpha trajectory plot for the IEEE paper.
├── stats_analysis.py            # Computes statistical significance (Wilcoxon tests) and confidence intervals.
│
├── data/                        # [Directory] Where the raw and pre-processed data lives before being fed to the AI.
│   ├── raw/                     # Original, untouched data files from external sources.
│   │   ├── supplygraph/         # Corporate supply chain dataset (Nodes, Edges, Temporal Data dirs).
│   │   ├── usgs/                # United States Geological Survey data (MCS2026 PDF, Extracted relationships, and Tables 1-7 CSVs).
│   │   └── synthetic/           # Empty placeholder for auto-generated test data.
│   └── processed/               # Data that has been cleaned, normalized, and cached by the loaders to save time on future runs.
│
├── results/                     # [Directory] Where training outputs, evaluations, and visualizations are stored.
│   ├── fig1_architecture.png    # Architecture diagram of the ATSF model.
│   ├── fig2_alpha_trajectory.*  # Generated plots tracking the fusion weight (alpha) over time.
│   ├── fig_edge_attention.*     # Generated visualizations of GAT edge attention weights.
│   ├── gen_edge_attention_fig.py# Script to generate the edge attention visual plot.
│   ├── reproducibility_report.json # Automated check proving identical deterministic outputs.
│   ├── Wilcoxon test.txt        # Statistical significance outputs comparing results against baselines.
│   │
│   ├── supplygraph/             # Results specific to the corporate supply chain dataset.
│   │   ├── checkpoints/         # Saved 'brains' (.pt files) of the trained model (e.g., best_model_seed42.pt).
│   │   ├── logs/                # Training histories saved as JSON (loss curves, learning rates).
│   │   ├── plots/               # Visualizations: Risk Dashboards, Prediction Line-charts, Alpha Distributions.
│   │   │   ├── alpha_by_epoch.png  # Tracks how the fusion weight changes across different training seeds.
│   │   │   ├── model_comparison.png# Bar charts comparing the primary metrics (MAE/RMSE) against baselines.
│   │   │   └── training_curves.png # Visualizes the train vs validation loss drop over epochs.
│   │   ├── multi_seed_metrics.csv # Tabular results of 5-seed stability tests.
│   │   ├── training_results.json# High-level summary of the training time and final loss.
│   │   └── classical_baselines_supplygraph.json # Benchmark scores of ARIMA and XGBoost.
│   │
│   └── usgs/                    # Results specific to the global mineral supply dataset.
│       ├── checkpoints/         # Saved 'brains' (.pt files) of the trained model.
│       ├── logs/                # Training histories saved as JSON (loss curves, learning rates).
│       ├── plots/               # Visualizations for the USGS dataset predictions.
│       │   ├── alpha_by_epoch.png  # Tracks the USGS fusion weight variations over epochs.
│       │   ├── model_comparison.png# Comparison of USGS errors against simple baselines.
│       │   └── training_curves.png # Loss convergence tracks for USGS data.
│       ├── multi_seed_metrics.csv # Tabular results of 5-seed stability tests for USGS.
│       ├── training_results.json# High-level summary of USGS training runs.
│       ├── classical_baselines_usgs.json # Benchmark scores of non-learned baselines (ARIMA/XGBoost).
│       ├── edge_type_analysis.json # Detailed output of which supply chain edge types are attended to most.
│       ├── edge_type_importance.csv # Extracted attention weights across different structural edge types.
│       └── edge_type_ablation.csv # Results showing model performance when specific edge types are removed.
│
├── src/                         # [Directory] The core engine and raw code of the project.
│   ├── models/                  # [Directory] The physical structure of the AI's Brain.
│   │   ├── complete_model.py    # The Grand Orchestrator: Wires all the other brain modules together.
│   │   ├── tft_branch.py        # The Time Traveler (Layer 2a): Analyzes historical timestamps purely.
│   │   ├── gat_branch.py        # The Web Weaver (Layer 2b): Analyzes the structural supply chain map purely.
│   │   ├── fusion_layer.py      # The Supreme Judge (Layer 3): Intelligently blends the Time and Map data.
│   │   ├── fused_representation.py # The Translator (Layer 4): Turns raw math into concrete demand numbers.
│   │   ├── risk_decision_layer.py  # The Safety Officer (Layer 5): Turns forecasts into actionable business risk warnings.
│   │   └── __init__.py          # Empty file telling Python this folder is a package.
│   │
│   ├── data/                    # [Directory] The data handlers that feed the model.
│   │   ├── graph_builder.py     # The Map Maker: Draws mathematical connection lines between products.
│   │   ├── supplygraph_loader.py# The Corporate Librarian: Normalizes and loads corporate supply chain CSVs into tensors.
│   │   ├── usgs_loader.py       # The Global Librarian: Normalizes and loads government mineral data.
│   │   ├── usgs_preprocessing.py# The Data Janitor: Cleans up messy raw government text PDFs into pristine data.
│   │   └── __init__.py          # Empty package file.
│   │
│   ├── training/                # [Directory] Tools to teach the AI.
│   │   ├── loss.py              # The Teacher's Red Pen: Mathematical formulas to measure how wrong the AI is during training.
│   │   ├── metrics.py           # The Scoreboard: Final presentation metrics (RMSE, MAE, Percentage Errors).
│   │   ├── trainer.py           # The Coach: The main loop that forces the model to practice millions of times.
│   │   └── __init__.py          # Empty package file.
│   │
│   └── utils/                   # [Directory] The Toolbelt (shared utilities).
│       ├── config.py            # Converts the config.yaml blueprint into Python code.
│       ├── seed.py              # Locks down system randomness so experiments are 100% reproducible.
│       ├── visualization.py     # The Artist: Draws all the charts, graphs, and distribution diagrams.
│       └── __init__.py          # Empty package file.
│
└── experiments/                 # [Directory] Where we actually start, test, and analyze the engine.
    ├── config.yaml              # The Master Blueprints: ALL settings (batch sizes, neuron counts, learning rates) live here.
    ├── train.py                 # The "ON" Switch: The main script you run to start training a new AI brain from scratch.
    ├── evaluate.py              # The Final Exam: Evaluates our model against simplified models on unseen data.
    ├── run_multi_seed.py        # The Scientific Method: Runs the entire train/evaluate process 5 times to mathematically prove our results aren't lucky flukes.
    ├── run_classical_baselines.py # The Old Rivals (Math): Tests our AI against older statistical math models like ARIMA and XGBoost.
    ├── run_graph_baselines.py   # The Old Rivals (Graphs): Tests our AI against standard non-temporal network models like GraphSAGE.
    ├── demo_risk_scoring.py     # Business Output: Generates official Risk Assessment JSON reports for executives.
    ├── edge_type_analysis.py    # Diagnostic tool to see which supply chain connections the AI trusts the most.
    ├── check_china.py           # Highly specific script checking the model's behavior regarding Chinese mineral production.
    ├── system_dry_run.py        # 10-second ultra-fast test run to make sure the code doesn't crash before leaving it overnight.
    ├── verify_reproducibility.py# Strict test proving that running the code twice gives identical results to the 6th decimal place.
    ├── verify_data.py           # QA check: Are there missing values or broken tensors in the SupplyGraph dataset?
    ├── verify_usgs_integrity.py # QA check: Are there missing values or broken tensors in the USGS dataset?
    ├── test_data_loading.py     # Simple unit test for SupplyGraph loading mechanics.
    └── test_usgs_loading.py     # Simple unit test for USGS loading mechanics.
```
