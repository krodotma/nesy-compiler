export interface TemplateSlot {
  name: string;
  required: boolean;
  defaultValue?: string;
}

/**
 * Template-based prompt construction
 */
export class PromptTemplate {
  private template: string;
  private slots: TemplateSlot[];

  constructor(template: string, slots: TemplateSlot[] = []) {
    this.template = template;
    this.slots = slots;
  }

  render(values: Record<string, string>): string {
    let result = this.template;

    for (const slot of this.slots) {
      const value = values[slot.name] ?? slot.defaultValue;
      if (slot.required && value === undefined) {
        throw new Error(`Missing required slot: ${slot.name}`);
      }
      if (value !== undefined) {
        result = result.replaceAll(`{{${slot.name}}}`, value);
      }
    }

    return result;
  }

  getSlots(): TemplateSlot[] {
    return [...this.slots];
  }
}
