"""Plain-language labels for the 20 features.

The datasets ship WITHOUT a data dictionary. These descriptions are generic readings of the
column names and are not official definitions. All features are integer scores (observed 0-22).
"""

FEATURE_DESCRIPTIONS = {
    "MonsoonIntensity": "Score related to monsoon (seasonal rainfall) intensity",
    "TopographyDrainage": "Score related to how terrain affects drainage",
    "RiverManagement": "Score related to river management",
    "Deforestation": "Score related to deforestation",
    "Urbanization": "Score related to urbanization",
    "ClimateChange": "Score related to climate-change influence",
    "DamsQuality": "Score related to the quality of dams",
    "Siltation": "Score related to siltation (sediment build-up)",
    "AgriculturalPractices": "Score related to agricultural practices",
    "Encroachments": "Score related to encroachment on waterways / flood plains",
    "IneffectiveDisasterPreparedness": "Score related to ineffective disaster preparedness",
    "DrainageSystems": "Score related to drainage systems",
    "CoastalVulnerability": "Score related to coastal vulnerability",
    "Landslides": "Score related to landslides",
    "Watersheds": "Score related to watersheds",
    "DeterioratingInfrastructure": "Score related to deteriorating infrastructure",
    "PopulationScore": "Score related to population",
    "WetlandLoss": "Score related to wetland loss",
    "InadequatePlanning": "Score related to inadequate planning",
    "PoliticalFactors": "Score related to political factors",
}
