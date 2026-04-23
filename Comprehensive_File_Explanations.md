
# Comprehensive File Explanations: Demand Forecasting Project

This guide provides professor-level, deeply intuitive explanations of every core file in this project. Structured so that anyone—even a beginner—can profoundly understand the "what," "why," and "how" of each script.

---

## `src/models/complete_model.py`
**The Grand Orchestrator**

Imagine you are building a robot to predict the future (in this case, supply chain demand). You can't just slap a brain in a box; you need eyes to see the past (time), a map to understand relationships (structure), a judge to weigh the evidence (fusion), a translator to speak the final prediction (representation), and a safety officer to warn about risks (risk scoring).

`complete_model.py` is the **body of the robot**. It doesn't do the deep thinking itself, but it takes all the specialized modules and plugs them together in the right order.

- **Layer 2a (TFT Branch - Temporal):** The component that looks at historical data over time.
- **Layer 2b (GAT Branch - Structural):** The component that looks at how different products and factories connect.
- **Layer 3 (Adaptive Fusion):** The judge that decides whether to trust the timeline (TFT) or the map (GAT) more.
- **Layer 4 (Fused Representation):** Turns the fused brainwaves into actual numbers (e.g., "We need 500 units next month").
- **Layer 5 (Risk Scoring):** Checks if the prediction might bankrupt us or cause severe shortages.

**Core Concept for Students:** This file defines the `AdaptiveFusionForecaster` class. Its literal job is to take the input data (the `batch`), pass it sequentially through the sub-networks (TFT -> GAT -> Fusion -> Output -> Risk), and return a neatly packaged dictionary of predictions and insights. It's the ultimate conductor of our neural network orchestra.

---

## `src/models/tft_branch.py`
**The Time Traveler (Layer 2a)**

The **TFT (Temporal Fusion Transformer)** is a highly advanced machine designed to look at the past to predict the future, while explaining *why* it made its choices. If standard neural networks are "black boxes," the TFT is a "glass box."

Here is how it breaks down complex timelines:
1. **Variable Selection Network (VSN):** Like a filter. If you're predicting umbrella sales, it learns to pay attention to "rainfall" and ignore "the color of the delivery truck." It explicitly learns which features matter.
2. **LSTM Encoder:** A type of memory. It reads the sequence of events over time from left to right, building a "story" of what happened up to the present day.
3. **Interpretable Multi-Head Attention:** It acts like a spotlight. Instead of treating all past days equally, it might say, "To predict next December's sales, I need to shine my spotlight directly on last December's data."
4. **Quantile Outputs:** It doesn't just guess one number. It gives a "best case," "worst case," and "most likely case" (quantiles) so we know how confident it is.

**Core Concept for Students:** `tft_branch.py` is essentially a hyper-smart trend analyzer. It encodes historical sequences into a profound mathematical representation called a "temporal embedding." It allows the model to understand the *time-based rhythms* of demand.

---

## `src/models/gat_branch.py`
**The Web Weaver (Layer 2b)**

While the TFT looks backward in time, the **GAT (Graph Attention Network)** looks *sideways* across relationships. 

Imagine a supply chain: A factory in Ohio supplies two warehouses, which supply fifty stores. If the Ohio factory shuts down, all fifty stores suffer. How does an AI learn this domino effect? Through a graph!

- **Nodes:** The entities (factories, products, warehouses).
- **Edges:** The connections between them.
- **Edge-Type Awareness:** Not all connections are equal. "Supplies to" is different from "Is an alternative product to." The `EdgeTypeAwareGAT` assigns different mathematical weights and meanings depending on the *type* of relationship.
- **Attention Mechanism:** Like a gossip network. A node (warehouse) "listens" to the information from its neighbors (factories and stores). If a factory says "I'm running low," the warehouse node updates its own state to reflect incoming panic. The "Attention" part means it learns which neighbors to listen to most closely.

**Core Concept for Students:** `gat_branch.py` processes the network of relationships. It produces a "structural embedding" for every product, meaning the mathematical representation of a product now contains information about its entire interconnected family tree. It allows the model to understand *spatial or relational* risks.

---

## `src/models/fusion_layer.py`
**The Supreme Judge (Layer 3: Adaptive Fusion)**

Imagine you have two advisors. Advisor A (TFT) perfectly remembers history and says, "Sales always drop in winter." Advisor B (GAT) looks at the map and says, "But our main competitor just went bankrupt in this region, so sales will spike!" Who do you trust?

The **Adaptive Fusion Layer** is the judge that decides. This is the **core, unique innovation of this entire project**. It doesn't just average the two advisors. It looks at the specific *material type* and the *forecast horizon*. 

- If we are predicting 1 month ahead (short-term), historical trends (TFT) might be more reliable.
- If we are predicting 12 months ahead (long-term), structural supply-chain shifts (GAT) might dominate.
- It dynamically calculates a weight called **Alpha** between 0 and 1. 
  - `Alpha = 1.0` means 100% trust the timeline (TFT).
  - `Alpha = 0.0` means 100% trust the graph (GAT).

**Core Concept for Students:** `fusion_layer.py` ensures the model isn't rigid. It dynamically adapts its "brain structure" specifically for the data it's looking at right this second. It outputs the combined "fused embedding" and the `alpha` value (so we humans can see exactly *why* it made its choice).

---

## `src/models/fused_representation.py`
**The Translator (Layer 4)**

We now have a brilliant, fused mathematical representation of the data. But the business executives don't speak "64-dimensional tensor math." They speak "How many units do we need?"

The **Fused Demand Representation** translates the complex math back into reality.
1. **Point Forecast:** The best single guess for demand (e.g., 500 units).
2. **Uncertainty Estimates:** The model admits what it doesn't know. It calculates a Standard Deviation.
3. **Confidence Intervals:** It provides a "Lower Bound" (e.g., 450 units) and "Upper Bound" (e.g., 550 units).

**Core Concept for Students:** `fused_representation.py` turns abstract machine learning embeddings into actual, usable supply chain numbers that tell us what to expect and how confident the AI is about that expectation.

---

## `src/models/risk_decision_layer.py`
**The Safety Officer (Layer 5)**

Predicting demand is only half the battle. If we predict we need 10,000 laptops, but we only have a budget for 5,000, we have a crisis. 

This layer explicitly translates the mathematical forecast into **human, business-level alarms.**
1. **Budget Stress:** (Expected Cost) divided by (Our Budget). If this is > 1.0, we get a "Critical" warning!
2. **Lead-time Risk:** Do we need the items *before* they can physically arrive? If yes, ring the alarm.
3. **Dependency Criticality:** If everyone relies on one specific supplier (found via GAT Attention), and that supplier fails, the whole chain collapses.

**Core Concept for Students:** `risk_decision_layer.py` takes the cold, hard numbers and applies human business rules to them. It literally spits out text actions like "[URGENT] Expedite supplier - Stock insufficient for lead time" so managers know exactly what to do.

---

## `src/data/graph_builder.py`
**The Map Maker**

If the AI needs to understand relationships, someone has to build the map. The `graph_builder.py` takes raw business data (e.g., "Product A is made in Plant X", "Product B is made in Plant X") and draws lines (edges) between them.

It creates a "Graph" where:
- Nodes = Products.
- Edges = Connections (like "Same Plant", "Same Product Group", "Same Storage").

**Core Concept for Students:** This script mathematically proves that "no product is an island." It turns a simple list of products into an interconnected web, allowing the model to see how a disruption in one area ripples through the entire network.

---

## `src/data/supplygraph_loader.py`
**The Librarian (Real-World Supply Chain)**

Deep learning models are incredibly picky eaters; they only eat tensors (multi-dimensional grids of numbers). They cannot read Excel files or CSVs directly.

The `SupplyGraphLoader`'s job is to read the real-world **SupplyGraph dataset** (a famous Kaggle dataset) and carefully process it into perfectly formatted tensors.
1. It reads the Nodes (products).
2. It reads the Edges (the map built above).
3. It reads the Temporal Data (Daily Sales Orders, Production).
4. **Crucial Step:** It normalizes the data. It uses "Z-score normalization" so a product that sells 1 unit/day and a product that sells 1,000 units/day are mathematically treated fairly by the AI.

**Core Concept for Students:** This is the bridge between raw messy CSV files and the ultra-precise AI. It meticulously organizes the data into sliding windows (e.g., "Take the last 30 days to predict the next 1, 3, 6, and 12 days").

---

## `src/data/usgs_loader.py`
**The Global Librarian (Minerals & Geopolitics)**

While the previous file handled standard supply chains, this file handles the **USGS (United States Geological Survey) Mineral Commodity dataset**.

Minerals are different from normal products. This data is:
- **Yearly** (not daily).
- **Global** (not just one company's factories).
- **Highly Complex:** It tracks 15 different types of relationships (e.g., "Is a byproduct of", "Price correlation", "Geopolitical supply risk").

**Core Concept for Students:** This file proves the model's flexibility. By simply swapping the `SupplyGraphLoader` for the `USGSLoader`, our AI transforms from a "corporate supply chain predictor" into a "global geopolitical mineral risk analyzer."

---

## `src/data/usgs_preprocessing.py`
**The Data Janitor**

The raw data from the US government is notoriously messy. It's meant for humans to read in PDFs, not for computers to process.

This script does the dirty work:
1. **Node Mapping:** Standardizes the names of 127 different minerals.
2. **Edge Processing:** Extracts text like "Cobalt is a byproduct of Copper" and turns it into a mathematical link (an edge).
3. **Feature Extraction:** Pulls out 12 key signals like "Imports", "Exports", "World Production".
4. **Static Features:** Scrapes advanced tables to find out if a mineral is heavily reliant on imports (Net Import Reliance) or if there's a risk of an export ban.

**Core Concept for Students:** AI is only as good as its data. This script embodies the phrase "Garbage In, Gold Out" by cleaning up messy government reports into pristine, structured data for the `USGSLoader`.

---

## `src/training/loss.py`
**The Teacher's Red Pen**

How does an AI learn? It makes a guess, compares it to reality, and measures how wrong it was. The "Loss" is the mathematical measurement of "wrongness."

This file contains the custom ways we measure wrongness:
1. **QuantileLoss:** Instead of just saying "you missed the exact number," it penalizes the model differently for predicting too high vs too low. This is how the model creates its Confidence Intervals (the "best case" and "worst case" bounds).
2. **AdaptiveFusionLoss:** This is the ultimate test score. It combines the forecasting error (were the predictions accurate?) with an "Alpha Regularization" penalty. It forces the model to actually *use* its decision-making judge (Layer 3), rather than just lazily trusting only the time data or only the graph data every time.

**Core Concept for Students:** `loss.py` is the grading rubric. By changing how we grade the model during training, we force it to develop specific behaviors (like estimating its own uncertainty and balancing its two brains).

---

## `src/training/metrics.py`
**The Scoreboard**

While `loss.py` is how the *model* judges itself during training, `metrics.py` is how *we* judge the model after it's done. 

It calculates standard human-readable statistics:
- **MAE (Mean Absolute Error):** On average, how many units were we off by?
- **RMSE (Root Mean Squared Error):** Penalizes massive mistakes heavily.
- **WAPE / SMAPE (Percentage Errors):** Are we off by 5% or 50%? This is crucial because being off by 10 units matters a lot for an item that sells 20 a day, but doesn't matter for an item that sells 5,000 a day.

**Core Concept for Students:** These are the final grades on the AI's report card that we show to the business executives. 

---

## `src/training/trainer.py`
**The Coach (Training Loop)**

The `ReproducibleTrainer` is the personal trainer making the AI pump iron. 

It handles the repetitive loop of machine learning:
1. Feeds a batch of data to the model.
2. Gets the predictions.
3. Uses `loss.py` to calculate the error.
4. Uses `optimizer.step()` to mathematically adjust the model's brain to be slightly better next time.
5. Repeats this thousands of times.

It also includes:
- **Early Stopping:** If the model stops improving, the Coach stops the workout so the model doesn't over-train (memorize the data).
- **Seed Management:** Ensures that if we run the exact same experiment twice, we get the *exact* same mathematical results. 

**Core Concept for Students:** `trainer.py` turns the static blueprint of the Neural Network into a living, learning entity. 

---

## `experiments/train.py`
**The Master Control Room**

This script is the main entry point to start the AI training process. You run this when you want to build a new brain from scratch.

It does the following carefully orchestrated steps:
1. **Reads Instructions:** Loads the instructions from `config.yaml`.
2. **Prepares Data:** Asks either the `SupplyGraphLoader` or `USGSLoader` for the training data.
3. **Builds the Robot:** Initializes the `AdaptiveFusionForecaster`.
4. **Starts the Workout:** Hands the model and the data over to the `ReproducibleTrainer`.
5. **Saves the Brain:** Once training is done, it saves the best version of the model's brain (`best_model_seed42.pt`) to the hard drive so it can be used later.

**Core Concept for Students:** This is the "ON" switch for the whole project. It wires all the pieces together (Data -> Model -> Trainer) and sets the machine in motion.

---

## `experiments/evaluate.py`
**The Final Exam**

After the model is trained, we need to know how good it actually is. We can't just trust the model's training scores; doing so would be like letting a student grade their own homework. We must test it on data it has *never* seen before (the Test Set).

This file runs an **Ablation Study**. It doesn't just evaluate our final model; it evaluates 4 different versions of it:
1. **TFT-only:** What if the model only looks at Time?
2. **GAT-only:** What if the model only looks at the Supply Chain Graph?
3. **Fixed Fusion:** What if the model just automatically splits the difference 50/50?
4. **Adaptive Fusion (Ours):** What if the model gets to intelligently choose what to look at?

**Core Concept for Students:** This is where we prove our hypothesis. By comparing our "Adaptive" model against the simpler versions, we mathematically prove that our Layer 3 (The Supreme Judge) actually works and makes the model smarter.

---

## `experiments/run_multi_seed.py`
**The Scientific Method (Repeatability)**

In machine learning, there is a lot of inherent randomness (how weights are initialized, how data is shuffled). Sometimes, a model gets a great score purely by lucky random chance.

To be rigorously scientific, this script forces the `train.py` and `evaluate.py` process to run **5 complete times**, starting from completely different random starting points (seeds).

**Core Concept for Students:** This script proves that our model isn't just a "one-hit wonder." It calculates the average performance across 5 different lives, providing a true, rock-solid evaluation metric (Mean ± Standard Deviation) required for publishing research papers.

---

## `experiments/run_classical_baselines.py` & `run_graph_baselines.py`
**The Old Rivals**

To prove your new AI is good, you must compare it against the old ways of doing things.
- **`run_classical_baselines.py`**: Tests against old-school statistical math models (ARIMA) and traditional machine learning (XGBoost).
- **`run_graph_baselines.py`**: Tests against a standard network algorithm (GraphSAGE) that has no concept of time whatsoever.

**Core Concept for Students:** This is the baseline. If our ultra-complex, multi-layered Deep Learning model can't beat older, simpler techniques, then we wasted our time. This file generates the benchmark scores that our new AI must defeat.

---

## `experiments/config.yaml`
**The Blueprints**

This isn't Python code; it's a configuration file. Instead of hardcoding numbers into our Python scripts, we put them all here.

It contains:
- **Data Settings:** Use the SupplyGraph dataset vs the USGS dataset.
- **Model Settings:** How "wide" the neural network is (e.g., `hidden_dim: 128`), how many layers it has.
- **Training Settings:** The Learning Rate, Batch Size, and the maximum number of times to loop over the data (epochs).

**Core Concept for Students:** By separating the numbers from the code, you can run entirely different experiments (e.g., making the model twice as large to see if it learns better) just by changing text in this one file, without touching a single line of Python.

---

## Utility Scripts in `experiments/` (The QA Department)
There are several smaller scripts in the experiments folder devoted entirely to **Quality Assurance** and sanity checks:

*   **`verify_data.py`** & **`verify_usgs_integrity.py`**: Did the data load correctly? Are there missing values? Are the math tensors the exact shapes we expect?
*   **`system_dry_run.py`**: A 10-second ultra-fast test run. It pushes 1 tiny batch of data through the whole system just to make sure the code doesn't crash before you leave an experiment running overnight.
*   **`verify_reproducibility.py`**: A strict test that runs the model twice with the *exact* same random seed, mathematically proving that run #1 and run #2 produce the exact same final error up to the 6th decimal place.
*   **`demo_risk_scoring.py`**: Runs Layer 5 to generate a business-friendly Risk Report and saves it as a JSON file, showing the practical, real-world output of the AI.

**Core Concept for Students:** Good engineering isn't just about building the core engine; it's about building all the testing tools *around* the engine to prove it operates safely and predictably.

---

## `src/utils`
**The Toolbelt**

Every large project needs small, reusable tools that everyone else shares.
- **`config.py`**: A tiny script that reads the `experiments/config.yaml` file and turns it into a Python dictionary.
- **`seed.py`**: The "Reproducibility Engine." Computers aren't actually random; they use math to fake it. This script locks down that math (setting the "seed" to a specific number like `42`) so that if we run an experiment on Monday, we get the exact same results if we run it again on Friday. This guarantees our research is scientifically sound.
- **`visualization.py`**: The "Artist." It takes the raw, massive walls of numbers outputted by the model and turns them into beautiful, publication-ready charts (Training Curves, Confidence Intervals, Distribution Plots, and the Risk Dashboard).

**Core Concept for Students:** Code isn't just about AI algorithms. Half of professional software engineering is building the plumbing (utils) that makes the algorithm easy to configure, easy to reproduce, and easy to visualize.

---

# Summary of the Entire System

1. **The Data Layer:** `[dataset]_loader.py` and `usgs_preprocessing.py` clean up messy real-world data and convert it into tensors. `graph_builder.py` mapped out the relationships.
2. **The Components (Layer 2):** `tft_branch.py` looks purely at historical timelines. `gat_branch.py` looks purely at structural blueprints.
3. **The Core Innovation (Layer 3):** `fusion_layer.py` acts as the Supreme Judge, intelligently weighting when to trust the timeline vs when to trust the blueprint.
4. **The Output (Layer 4 & 5):** `fused_representation.py` spits out the mathematical forecast with confidence intervals. `risk_decision_layer.py` converts that math into human-readable business warnings.
5. **The Control Center:** `complete_model.py` wires the layers together. `train.py` trains the brain. `evaluate.py` proves it works. `visualization.py` paints the picture.

## Additional Experiment & Analysis Scripts
During verification, we found a few smaller scripts used for specific analysis and testing:

*   **`experiments/check_china.py`**: A highly specific script that checks the model's behavior regarding China's mineral production data within the USGS dataset, likely related to geopolitical supply risk analysis.
*   **`experiments/edge_type_analysis.py`**: A diagnostic tool that analyzes the different specific relationships (edges) in the Graph Attention Network to see which relationships the model relies on the most.
*   **`experiments/test_data_loading.py`** & **`test_usgs_loading.py`**: Simple unit tests that developers use to verify the loading mechanics of the SupplyGraph and USGS datasets work perfectly without having to run the entire massive `train.py` script.

## The `__init__.py` Files
Across the project in folders like `src/`, `src/models/`, `src/training/`, and `src/utils/`, you will see files named `__init__.py`. 
These are completely empty files. In Python, placing an empty `__init__.py` file inside a folder tells the computer: *"Treat this folder as an official Python Package, so other scripts can import code from it."* Without them, the folders are just dead directories.

# Data & Results Folders Breakdown

## The `data/` Directory (The Raw Ingredients)
Before the AI can do any forecasting, it needs food. That food is stored here.

### `data/raw/supplygraph/`
This contains the Kaggle-style corporate supply chain dataset, heavily structured:
- **`Nodes/`**: Contains the CSV files listing the actual physical products, factories, and storage locations.
- **`Edges/`**: Contains CSV files listing how the products move from factory to factory.
- **`Temporal Data/`**: The daily sales order history (the sequence of numbers the model will try to predict).

### `data/raw/usgs/`
This is the messy, real-world data from the US Government regarding geopolitics and minerals:
- **`mcs2026.pdf`**: The massive 200+ page text report authored by the government.
- **`MCS2026_*.csv`**: Various tables (Tables 1 through 7, Figures 1 through 13) extracted from the PDF detailing Net Import Reliance, Major Import Sources (like China or Canada), Trade Agreements, and Global Production.
- **`pdf_extracted_relationships.csv`**: A map generated by the Janitor script showing which minerals are co-mined with each other or used in the same industries.

### `data/processed/`
A cache folder. When the loaders finish spending 3 minutes cleaning the `raw` data, they save the final pristine PyTorch Tensors here. Next time the model runs, it just loads from here in 2 seconds.

---

## The `results/` Directory (The Trophies and Records)
When an experiment finishes, the output is sorted into dataset-specific folders (`results/supplygraph/` and `results/usgs/`) so you don't overwrite your corporate results with your mineral results.

Inside each of these folders, you will find:
- **`checkpoints/`**: The `.pt` (PyTorch) files. This is literally the AI's "brain" saved to your hard drive. It contains the millions of decimal numbers (weights) the model learned. If you load this file, the model remembers everything it was taught.
- **`logs/`**: `.json` files that track the model's error rate every single epoch during training. This proves the model actually improved over time.
- **`plots/`**: The `.png` files generated by the Artist (`visualization.py`). You will find the Model Comparison bar charts, the Risk Dashboard, and the Alpha Distribution graphs here.
- **`multi_seed_metrics.csv`**: An Excel-friendly file proving the AI is stable across 5 different random seeds. 
- **`classical_baselines_*.json`**: The benchmark test scores from the older mathematical models.


# Deep Dive into Missing Subfolders (Data & Results)

Based on a complete recursive scan, there are many detailed files inside exactly how the raw data and results are structured:

## Deep Dive: `data/raw/` and `data/processed/`

### 1. `data/raw/supplygraph/`
*   **`Nodes/`**: `Nodes.csv` (the main entites), `Node Types (Product Group and Subgroup).csv` (categorizations), and `NodesIndex.csv`.
*   **`Edges/`**: Contains relational maps defining `Edges (Plant).csv`, `Edges (Product Group).csv`, `Edges (Product Sub-Group).csv`, and `Edges (Storage Location).csv`. It also includes an `EdgesIndex/` subfolder holding index mappings.
*   **`Temporal Data/`**: The time-series sequences. Split by metric (e.g., `Unit/`, `Weight/`) measuring `Delivery To distributor.csv`, `Factory Issue.csv`, `Production .csv`, and `Sales Order.csv`.

### 2. `data/raw/usgs/`
*   **`mcs2026.pdf`**: The full text report.
*   **`MCS2026_Commodities_Data.csv`**: The massive 3MB dataset housing the bulk of the mineral statistics.
*   **`MCS2026_Mineral_Industry_Trends_and_Salient_Statistics/`**: A subfolder where all 13 Figures and 7 Tables from the report are saved as individual CSV files (e.g., `MCS2026_Fig10_Price_Growth_Rates.csv`, `MCS2026_T6_Critical_Minerals_End_Use.csv`).

### 3. `data/processed/`
*   **`data/processed/usgs/`**: When `usgs_preprocessing.py` finishes parsing everything, it generates pristine standard datasets saved here. This includes the individual metrics (`Consumption.csv`, `Production.csv`, `Imports.csv`, `Price.csv`) and the complex relationship edges (e.g., `Edges (battery_coproduction).csv`, `Edges (geopolitical_supply_risk).csv`, `Edges (substitution).csv`).

---

## Deep Dive: `results/` 

The `results/` folder tracks three different test runs: `supplygraph`, `supplygraph_val`, and `usgs`.

### 1. Checkpoints (`results/*/checkpoints/`)
*   Contains the brains from the multi-seed runs: `best_model_seed42.pt`, `best_model_seed123.pt`, `best_model_seed456.pt`, `best_model_seed789.pt`, and `best_model_seed1337.pt`. 
*   **`global_best_model.pt`**: Out of all 5 random seeds, the absolute best performing model across the whole sweep is duplicated and saved here.

### 2. Logs & JSON Tabulars (`results/*/logs/`)
*   **`logs/`**: Holds `training_history_seed[X].json` for all 5 seeds tracking loss curves per epoch.
*   **Root Results Folder**: 
    *   `multi_seed_metrics.csv`: The Excel file containing the aggregated Mean ± Standard Deviation metrics.
    *   `classical_baselines_*.json`: The scores scored by ARIMA/XGBoost.
    *   `training_results.json` & `evaluation_results.json`: Summary stats of the execution.

### 3. Plots (`results/*/plots/`)
*   Plots generated by the `ModelVisualizer`:
    *   `model_comparison.png`: The bar chart comparing TFT/GAT/Ours against baselines.
    *   `training_curves.png`: Loss curves graph.
    *   `alpha_by_epoch.png`: A line chart tracking how the Supreme Judge (Layer 3) changed its mind over time.
    *   `alpha_distribution_by_material.png`: Box plots showing which specific materials rely more on Graph vs Time.


# Exhaustive Tracked Files List
To ensure 100% coverage of every single file in the project directory, here are the highly specific remaining files and their exact purposes:

## Project Root Files
- **`.gitignore`**: Tells the Git version control system which files (like the `.pt` brains or `__pycache__`) to ignore and not upload to GitHub.
- **`README.md`**: The standard front-page instruction manual for the repository, describing the project's purpose and how to install its dependencies.

## Specific `data/raw/` Documentation Details
Inside the raw hierarchical data folders, there are additional specific trackers:
- **`data/raw/supplygraph/README.md`** & **`data/raw/supplygraph/Nodes/README.md`**: Specific readmes documenting Kaggle dataset schema intricacies.
- **`data/raw/supplygraph/Temporal Data/Weight/`**: Includes duplicates of the `Unit/` time-series data but measured by mass: `Delivery to Distributor.csv` and `Sales Order .csv`.
- **`data/raw/usgs/`** (and its subfolder `MCS2026_Mineral_Industry_Trends_and_Salient_Statistics/`): Houses the exact individual CSVs extracted: `MCS2026_Fig1_Minerals_in_Economy.csv`, `MCS2026_Fig2_Net_Import_Reliance.csv`, `MCS2026_Fig3_Major_Import_Sources.csv`, `MCS2026_Fig4_Value_by_Type.csv`, `MCS2026_Fig11_Pch_Consump_2024_2025.csv`, `MCS2026_Fig12_Pch_Consump_2021_2025.csv`, `MCS2026_Fig13_Scrap.csv`. As well as tables: `MCS2026_T1_Mineral_Industry_Trends.csv`, `MCS2026_T2_Mineral_Economic_Trends.csv`, `MCS2026_T3_State_Value_Rank.csv`, `MCS2026_T4_Country_Export_Control.csv`, `MCS2026_T5_Trade_Agreements.csv`, and `MCS2026_T7_Critical_Minerals_Salient.csv`.

## Specific `data/processed/usgs/` Edge & Node Files 
When the USGS data is successfully parsed, the janitor scripts build exact structural components:
- **Relationship Edges**: `all_edges.csv`, `Edges (alloy_components).csv`, `Edges (byproduct_coproduction).csv`, `Edges (catalyst_role).csv`, `Edges (construction_coproduction).csv`, `Edges (critical_mineral_designation).csv`, `Edges (electronics_coproduction).csv`, `Edges (functional_coating).csv`, `Edges (price_correlation).csv`, `Edges (recycling_secondary_source).csv`, `Edges (refractory_industrial).csv`, `Edges (supply_chain_input).csv`, and `Edges (technology_cluster).csv`. (These dictate exactly how minerals relate structurally).
- **Extracted Node Metrics**: `Consumption.csv`, `Employment.csv`, `Exports.csv`, `Node Types.csv`, `Sales.csv`, `StaticNodeFeatures.csv`, `Stocks.csv`, `Supply.csv`, `WorldCapacity.csv`, `WorldProduction.csv`, and `WorldReserves.csv`.
- **`processing_report.txt`**: A health-check text file generated after parsing concludes.

## Specific `results/` Tracking
- **Logs**: Specifically named exactly `training_history_seed42.json`, `training_history_seed123.json`, `training_history_seed456.json`, `training_history_seed789.json`, and `training_history_seed1337.json` for both `supplygraph` and `usgs` folders.
- **Specific CSVs/JSONs**: `results/supplygraph/classical_baselines_supplygraph.json`, `results/usgs/classical_baselines_usgs.json`, and `multi_seed_metrics_usgs.csv`.
- **Specific Plots**: `results/plots/model_comparison_supplygraph.png` and `results/plots/model_comparison_usgs.png`.
