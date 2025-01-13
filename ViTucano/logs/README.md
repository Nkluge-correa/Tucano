# Logs

This folder contains all logs related to the training of the ViTucano series. To read them, just use `pandas`:

```python
import pandas as pd

df = pd.read_parquet('./training-logs/training-logs-pretraining-1b5-v1.parquet')
```

## How do we interpret the output data from CodeCarbon?

After [significant investigation](https://github.com/mlco2/codecarbon/issues/544), we found out we need to:

- Get the measured `cpu_energy`, `gpu_energy`, and `ram_energy`.
We need to multiply `cpu_energy`, `gpu_energy`, and `ram_energy` by the number of nodes used (tracking apparently only works on single-node operations).

- To get the value from `gpu_energy`, you can multiply the medium wattage of all GPUS (W&B and the `nvidia-smi` can give you that, besides the logs from CodeCarbon) by the duration (in hours), and divide everything by 1000 to get kWh, i.e., `(duration * (wattage * n_gpus)) / 1000`.

- Hence, `energy_consumed` is acctualy `(cpu_energy + gpu_energy + ram_energy) * number_of_nodes`.
To get emissions, multiply the energy consumed (`energy_consumed * number_of_nodes`) by the carbon intensity of your current energy grid (e.g., Germany is 0.37 KgCO2.eq/KWh).

Here is how you can see the emmissions for the training (pretraing + fine-tuning) of the ViTucano models:

```python
import pandas as pd
pd.set_option('display.max_columns', None)

# Paths for the data
emission_paths = {
    '1b5-v1': "./emissions/emissions-1b5-v1.csv", # 1 node (A40)
    '2b8-v1': "./emissions/emissions-2b8-v1.csv", # 1 node (A40)
}
model = '2b8-v1'
carbon_intensity = 0.37
df = pd.read_csv(emission_paths[model])

energy_consumed_pretraining = 0
emissions_pretraining = 0
duration_pretraining = 0

energy_consumed_sft = 0
emissions_sft = 0
duration_sft = 0

for project_name in df.project_name.unique():
    if "pretraining" in project_name:
        df_run = df[df.project_name == project_name]
        duration_pretraining += df_run.iloc[-1]['duration']
        energy_consumed_pretraining += df_run.iloc[-1]['energy_consumed']
        emissions_pretraining += df_run.iloc[-1]['emissions']

    else:
        df_run = df[df.project_name == project_name]
        duration_sft += df_run.iloc[-1]['duration']
        energy_consumed_sft += df_run.iloc[-1]['energy_consumed']
        emissions_sft += df_run.iloc[-1]['emissions']

duration_pretraining = duration_pretraining / 3600
duration_sft = duration_sft / 3600

energy_consumed = energy_consumed_pretraining + energy_consumed_sft
emissions = emissions_pretraining + emissions_sft
duration = duration_pretraining + duration_sft

print(f"Duration Pretraining: {duration_pretraining:.2f} hours")
print(f"Energy consumed Pretraining: {energy_consumed_pretraining:.2f} kWh")
print(f"CO2 emitted Pretraining: {emissions_pretraining:.2f} KgCO2eq")
print("-" * 10)
print(f"Duration SFT: {duration_sft:.2f} hours")
print(f"Energy consumed SFT: {energy_consumed_sft:.2f} kWh")
print(f"CO2 emitted SFT: {emissions_sft:.2f} KgCO2eq")
print("-" * 10)
print(f"Total Duration: {duration:.2f} hours")
print(f"Total Energy consumed: {energy_consumed:.2f} kWh")
print(f"Total CO2 emitted: {emissions:.2f} KgCO2eq")
```
