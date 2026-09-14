import os

from azure.ai.ml import MLClient, command
from azure.identity import AzureCliCredential


required = [
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "AZURE_ML_WORKSPACE",
    "AZURE_ML_COMPUTE",
]

missing = [name for name in required if not os.getenv(name)]

if missing:
    raise RuntimeError(
        "Missing required environment variables: "
        + ", ".join(missing)
    )


credential = AzureCliCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_ML_WORKSPACE"],
)


environment_name = os.getenv(
    "AZURE_ML_ENVIRONMENT",
    "AzureML-acpt-pytorch-2.8-cuda12.6@latest",
)

experiment_name = os.getenv(
    "AZURE_ML_EXPERIMENT",
    "hello-gpu",
)


job = command(
    code="./src",
    command="python3 hello_gpu.py",
    environment=f"azureml:{environment_name}",
    compute=f"{os.environ['AZURE_ML_COMPUTE']}",
    experiment_name=experiment_name,
    display_name="hello-t4-gpu",
)


submitted = ml_client.jobs.create_or_update(job)

print()
print("Submitted AML job:")
print(f"  name:   {submitted.name}")
print(f"  status: {submitted.status}")
print()

print("Streaming job output...")
ml_client.jobs.stream(submitted.name)