#!/usr/bin/env python3
"""
Portal UI Mapper - Step 51, 52, & 54 of PORTAL Implementation.
Maps Agent Intents and Etymons to declarative A2UI components.
"""
import sys
import json

def get_component_catalog():
    return {
        "layout": "stack",
        "approved_components": [
            "Card", "Button", "Label", "ProgressBar", "CodeBlock", "LogicGraph"
        ]
    }

def map_intent_to_ui(orchestrator_result):
    mode = orchestrator_result.get("mode")
    decoded = orchestrator_result.get("decoded", {})
    
    # Step 54: Auto-generate labels
    etymons = decoded.get("etymons", [])
    primary_etymon = etymons[0] if etymons else "RAW_SIGNAL"
    
    # Step 52: Intent to Component Mapping
    a2ui_msg = {
        "type": "layout",
        "intent": f"PROCESS_{mode}",
        "payload": {
            "title": f"{primary_etymon} Entry Portal",
            "mode": mode,
            "components": [
                {
                    "type": "Label",
                    "props": {"text": f"Lineage: {mode}", "variant": "subtle"}
                },
                {
                    "type": "ProgressBar",
                    "props": {
                        "value": decoded.get("texture_density", 0) * 100,
                        "label": "Texture Density"
                    }
                },
                {
                    "type": "CodeBlock",
                    "props": {
                        "content": orchestrator_result.get("signal"),
                        "language": "text"
                    }
                }
            ]
        }
    }
    
    if mode == "AM":
        a2ui_msg["payload"]["components"].append({
            "type": "Button",
            "props": {"label": "Actualize Graft", "action": "actualize", "variant": "primary"}
        })
    else:
        a2ui_msg["payload"]["components"].append({
            "type": "Button",
            "props": {
                "label": "Promote to Actualized (Snap)", 
                "action": "promote_shadow", 
                "variant": "primary"
            }
        })
        a2ui_msg["payload"]["components"].append({
            "type": "Button",
            "props": {"label": "Keep in Shadow", "action": "shadow", "variant": "secondary"}
        })
        
    return a2ui_msg

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mock_res = json.loads(sys.argv[1])
        ui = map_intent_to_ui(mock_res)
        print(json.dumps(ui, indent=2))
