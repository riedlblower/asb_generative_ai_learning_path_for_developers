import warnings
warnings.filterwarnings('ignore')

import json
import os
import sys
from io import StringIO
import textwrap

import boto3
import botocore
from botocore.config import Config

from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate



def print_ww(*args, width: int = 100, **kwargs):
    """Like print(), but wraps output to `width` characters (default 100)"""
    buffer = StringIO()
    try:
        _stdout = sys.stdout
        sys.stdout = buffer
        print(*args, **kwargs)
        output = buffer.getvalue()
    finally:
        sys.stdout = _stdout
    for line in output.splitlines():
        print("\n".join(textwrap.wrap(line, width=width)))

def get_bedrock_client():
    service_name='bedrock-runtime'
    target_region = "us-west-2"
    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}
    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)
    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )
    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client

boto3_bedrock = get_bedrock_client() # configure_environment()

inference_modifier = {'max_tokens_to_sample':4096, 
                      "temperature":0.5,
                      "top_k":250,
                      "top_p":1,
                      "stop_sequences": ["\n\nHuman"]
                     }

textgen_llm = Bedrock(model_id = "anthropic.claude-v2",
                    client = boto3_bedrock, 
                    model_kwargs = inference_modifier 
                    )	
	

# Vehicle Fleet Management Code written in C++
sample_code = """
#include <iostream>
#include <string>
#include <vector>

class Vehicle {
protected:
    std::string registrationNumber;
    int milesTraveled;
    int lastMaintenanceMile;

public:
    Vehicle(std::string regNum) : registrationNumber(regNum), milesTraveled(0), lastMaintenanceMile(0) {}

    virtual void addMiles(int miles) {
        milesTraveled += miles;
    }

    virtual void performMaintenance() {
        lastMaintenanceMile = milesTraveled;
        std::cout << "Maintenance performed for vehicle: " << registrationNumber << std::endl;
    }

    virtual void checkMaintenanceDue() {
        if ((milesTraveled - lastMaintenanceMile) > 10000) {
            std::cout << "Vehicle: " << registrationNumber << " needs maintenance!" << std::endl;
        } else {
            std::cout << "No maintenance required for vehicle: " << registrationNumber << std::endl;
        }
    }

    virtual void displayDetails() = 0;

    ~Vehicle() {
        std::cout << "Destructor for Vehicle" << std::endl;
    }
};

class Truck : public Vehicle {
    int capacityInTons;

public:
    Truck(std::string regNum, int capacity) : Vehicle(regNum), capacityInTons(capacity) {}

    void displayDetails() override {
        std::cout << "Truck with Registration Number: " << registrationNumber << ", Capacity: " << capacityInTons << " tons." << std::endl;
    }
};

class Car : public Vehicle {
    std::string model;

public:
    Car(std::string regNum, std::string carModel) : Vehicle(regNum), model(carModel) {}

    void displayDetails() override {
        std::cout << "Car with Registration Number: " << registrationNumber << ", Model: " << model << "." << std::endl;
    }
};

int main() {
    std::vector<Vehicle*> fleet;

    fleet.push_back(new Truck("XYZ1234", 20));
    fleet.push_back(new Car("ABC9876", "Sedan"));

    for (auto vehicle : fleet) {
        vehicle->displayDetails();
        vehicle->addMiles(10500);
        vehicle->checkMaintenanceDue();
        vehicle->performMaintenance();
        vehicle->checkMaintenanceDue();
    }

    for (auto vehicle : fleet) {
        delete vehicle; 
    }

    return 0;
}
"""

from langchain.prompts import PromptTemplate

# Create a prompt template that has multiple input variables
multi_var_prompt = PromptTemplate(
    input_variables=["code", "srcProgrammingLanguage", "targetProgrammingLanguage"], 
    template="""

Human: You will be acting as an expert software developer in {srcProgrammingLanguage} and {targetProgrammingLanguage}. 
You will tranlslate below code from {srcProgrammingLanguage} to {targetProgrammingLanguage} while following coding best practices.
<code>
{code}
</code>

Assistant: """
)

# Pass in values to the input variables
prompt = multi_var_prompt.format(code=sample_code, srcProgrammingLanguage="C++", targetProgrammingLanguage="Java")

response = textgen_llm(prompt)

target_code = response[response.index('\n')+1:]

print_ww(target_code)




