import { GoogleGenAI, Type, Schema } from "@google/genai";

// Define strict interfaces instead of importing from potentially missing 'types'
export interface UserStats {
  height: number;
  weight?: number;
  gender?: 'male' | 'female' | 'other';
  age?: number;
  unitSystem: 'metric' | 'imperial';
}

export interface MeasurementResult {
  neck: number;
  shoulder: number;
  chest: number;
  waist: number;
  hips: number;
  inseam: number;
  sleeve: number;
  // ... partial list for brevity, the schema defines the full contract
  scaling_factor: number;
  estimated_height_cm: number;
  thought_summary: string;
  landmarks_front: any;
  landmarks_side: any;
  usage_metadata?: any;
  model_name?: string;
}

export class AnthropometristAgent {
  private ai: GoogleGenAI;
  private modelId: string;

  constructor(apiKey: string, modelId: string = "gemini-2.0-flash") {
    this.ai = new GoogleGenAI({ apiKey });
    this.modelId = modelId;
  }

  async analyze(
    frontImageBase64: string,
    sideImageBase64: string,
    stats: UserStats
  ): Promise<MeasurementResult> {
    
    const prompt = `
      You are an expert anthropometrist and technical tailor. 
      Analyze the attached TWO images (Front & Side) to calculate precise body measurements.
      
      GROUND TRUTH:
      - User Provided Height: ${stats.height} cm
      ${stats.weight ? `- Weight: ${stats.weight} kg` : ''}
      ${stats.gender ? `- Gender: ${stats.gender}` : ''}
      
      TASK:
      1. TRANSPARENCY & SCALING: 
         - Identify the top of the head and bottom of the feet in the Front image.
         - Calculate "scaling_factor" (pixels_per_cm) based on the User Provided Height vs the subject's pixel height.
         - Independently estimate the subject's height ("estimated_height_cm") based on head-to-body proportions to cross-check the user provided value.

      2. LANDMARKS:
         - Identify specific anatomical landmarks (x,y coordinates normalized 0-1) for both Front and Side views.
         - You MUST return "landmarks_front" and "landmarks_side" with the specific points defined in the schema.

      3. MEASUREMENT:
         - Measure raw pixel widths (Front) and depths (Side) for key areas (Chest, Waist, Hips, Thigh, etc.).
         - Apply geometric formulas (e.g., Ramanujan approximation for ellipse circumference) to convert pixel dimensions to cm circumferences using the scaling_factor.
      
      4. REASONING:
         - Provide a "thought_summary" (2-3 sentences) explaining how you determined the fit and any adjustments made for posture or clothing.

      5. QUALITY:
         - Assess image quality, pose, and lighting.
      
      Return strict JSON matching the provided schema.
    `;

    // Define the output schema strictly (Migrated from FT/services/geminiService.ts)
    const responseSchema: Schema = {
      type: Type.OBJECT,
      properties: {
        // Measurements
        neck: { type: Type.NUMBER },
        shoulder: { type: Type.NUMBER },
        chest: { type: Type.NUMBER },
        bicep: { type: Type.NUMBER },
        wrist: { type: Type.NUMBER },
        sleeve: { type: Type.NUMBER },
        waist: { type: Type.NUMBER },
        hips: { type: Type.NUMBER },
        inseam: { type: Type.NUMBER },
        outseam: { type: Type.NUMBER },
        thigh: { type: Type.NUMBER },
        calf: { type: Type.NUMBER },
        ankle: { type: Type.NUMBER },
        torso_length: { type: Type.NUMBER },

        // Transparency Fields
        scaling_factor: { type: Type.NUMBER, description: "Calculated pixels per cm" },
        estimated_height_cm: { type: Type.NUMBER, description: "AI estimated height based on proportions" },
        thought_summary: { type: Type.STRING, description: "Natural language summary of reasoning" },
        
        // Landmarks - Flattened Top Level
        landmarks_front: {
           type: Type.OBJECT,
           properties: {
             head_top: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
             neck_base: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
             shoulder_left: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
             shoulder_right: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
             waist_left: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
             waist_right: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
             hip_left: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
             hip_right: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
             knee_left: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
             knee_right: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
             ankle_left: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
             ankle_right: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
             feet_center: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} }
           }
        },
        landmarks_side: {
           type: Type.OBJECT,
           properties: {
              neck_point: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
              chest_front: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
              chest_back: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
              waist_front: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
              waist_back: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
              hip_front: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
              hip_back: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
              knee: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
              ankle: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} },
              back_spine: { type: Type.OBJECT, properties: {x: {type: Type.NUMBER}, y: {type: Type.NUMBER}} }
           }
        },

        confidence: { type: Type.NUMBER },
        notes: { type: Type.STRING },
        body_shape: { type: Type.STRING },
        
        quality_assessment: {
          type: Type.OBJECT,
          properties: {
            overall_score: { type: Type.NUMBER },
            issues_detected: { 
              type: Type.ARRAY, 
              items: { type: Type.STRING }
            }
          }
        },
      },
      required: [
        "neck", "shoulder", "chest", "waist", "hips", "inseam", 
        "confidence", "quality_assessment",
        "scaling_factor", "thought_summary", "landmarks_front", "landmarks_side"
      ],
    };

    const frontData = frontImageBase64.split(',')[1] || frontImageBase64;
    const sideData = sideImageBase64.split(',')[1] || sideImageBase64;

    const config: any = {
      responseMimeType: "application/json",
      responseSchema: responseSchema,
      temperature: 0.2,
    };

    // Apply Thinking Config ONLY for 2.5 Flash
    if (this.modelId.includes('gemini-2.5-flash')) {
      config.thinkingConfig = { thinkingBudget: 12000 }; 
    }

    const response = await this.ai.models.generateContent({
      model: this.modelId,
      contents: {
        parts: [
          { inlineData: { mimeType: "image/jpeg", data: frontData } },
          { inlineData: { mimeType: "image/jpeg", data: sideData } },
          { text: prompt }
        ]
      },
      config: config
    });

    if (!response.text) {
      throw new Error("No response text generated");
    }

    const result = JSON.parse(response.text) as MeasurementResult;
    result.model_name = this.modelId;
    return result;
  }
}
