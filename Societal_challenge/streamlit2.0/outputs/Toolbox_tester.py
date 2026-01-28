import yaml

with open("toolbox_solutions.yaml", "r") as f:
    toolbox = yaml.safe_load(f)

solutions = toolbox["solutions"]

def is_solution_applicable(cluster, solution):
    # 1. preconditions
    if "grid_constraints_present" in solution.get("preconditions", []):
        if not cluster["grid_constraints_present"]:
            return False, "Geen netcongestie"

    # 2. exclusion rules (hardcoded, v0)
    for rule in solution.get("exclusion_rules", []):
        condition = rule["condition"]

        if condition == "number_of_buildings < 5":
            if cluster["number_of_buildings"] < 5:
                return False, rule["reason"]

        if condition == "spatial_spread_m > 1000":
            if cluster["spatial_spread_m"] > 1000:
                return False, rule["reason"]

        if condition == "required_power_kw > 120":
            if cluster["required_power_kw"] > 120:
                return False, rule["reason"]

    return True, "Toepasbaar"


cluster = {
    "number_of_buildings": 3,
    "spatial_spread_m": 400,
    "required_power_kw": 200,
    "grid_constraints_present": True,
}

for solution in solutions:
    applicable, reason = is_solution_applicable(cluster, solution)
    print(f"{solution['name']}: {applicable} ({reason})")


