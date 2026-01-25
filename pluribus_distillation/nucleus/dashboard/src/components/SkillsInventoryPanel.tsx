import { component$, useSignal, useVisibleTask$, $ } from '@builder.io/qwik';

interface Skill {
  id: string;
  name: string;
  description: string;
  category: string;
}

interface SkillCategory {
  name: string;
  skills: Skill[];
}

export const SkillsInventoryPanel = component$(() => {
  const categories = useSignal<SkillCategory[]>([]);
  const loading = useSignal(true);
  const error = useSignal<string | null>(null);
  const filter = useSignal('');

  const fetchSkills = $(async () => {
    try {
      loading.value = true;
      error.value = null;
      // Fetch from file system API
      const res = await fetch('/api/fs/skills/SKILLS_MANIFEST.md');
      if (!res.ok) throw new Error(`Failed to load manifest: ${res.status}`);
      const text = await res.text();
      
      const parsed: SkillCategory[] = [];
      let currentCategory: SkillCategory | null = null;

      const lines = text.split('\n');
      for (const line of lines) {
        // Match Category: ## Category N: Name
        const catMatch = line.match(/^## Category \d+: (.+)$/);
        if (catMatch) {
          if (currentCategory) parsed.push(currentCategory);
          currentCategory = { name: catMatch[1].trim(), skills: [] };
          continue;
        }

        // Match Skill: N. **Name:** Description
        const skillMatch = line.match(/^\d+\.\s+\*\*(.+?):\*\*\s+(.+)$/);
        if (skillMatch && currentCategory) {
          currentCategory.skills.push({
            id: skillMatch[1].toLowerCase().replace(/\s+/g, '_'),
            name: skillMatch[1],
            description: skillMatch[2],
            category: currentCategory.name
          });
        }
      }
      if (currentCategory) parsed.push(currentCategory);
      categories.value = parsed;
    } catch (err) {
      error.value = String(err);
    } finally {
      loading.value = false;
    }
  });

  useVisibleTask$(() => {
    fetchSkills();
  });

  return (
    <div class="flex flex-col h-full bg-card rounded-lg border border-border overflow-hidden">
      {/* Header */}
      <div class="p-4 border-b border-border flex items-center justify-between shrink-0">
        <div class="flex items-center gap-3">
          <span class="text-2xl">🧠</span>
          <div>
            <h2 class="font-semibold text-foreground">Skills Inventory</h2>
            <p class="text-xs text-muted-foreground">177+ Elite & Cognitive Skills</p>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <input 
            type="text" 
            placeholder="Search skills..."
            class="px-3 py-1.5 text-xs bg-muted/50 border border-border rounded-md w-48 focus:outline-none focus:border-primary/50"
            value={filter.value}
            onInput$={(e) => filter.value = (e.target as HTMLInputElement).value}
          />
          <button 
            onClick$={fetchSkills}
            class="p-2 hover:bg-muted/50 rounded-md transition-colors"
            title="Refresh"
          >
            🔄
          </button>
        </div>
      </div>

      {/* Content */}
      <div class="flex-1 overflow-y-auto p-4 custom-scrollbar">
        {loading.value && (
          <div class="flex items-center justify-center h-32 text-muted-foreground">
            <div class="animate-spin mr-2">⟳</div> Loading manifest...
          </div>
        )}

        {error.value && (
          <div class="p-4 text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg">
            Error: {error.value}
          </div>
        )}

        {!loading.value && !error.value && (
          <div class="space-y-6">
            {categories.value.map((cat) => {
              const filteredSkills = cat.skills.filter(s => 
                !filter.value || 
                s.name.toLowerCase().includes(filter.value.toLowerCase()) || 
                s.description.toLowerCase().includes(filter.value.toLowerCase())
              );

              if (filteredSkills.length === 0) return null;

              return (
                <div key={cat.name} class="space-y-3">
                  <h3 class="text-sm font-semibold text-primary/80 sticky top-0 bg-card/95 backdrop-blur py-2 z-10 border-b border-border/50">
                    {cat.name}
                    <span class="ml-2 text-xs text-muted-foreground font-normal">({filteredSkills.length})</span>
                  </h3>
                  
                  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                    {filteredSkills.map((skill) => (
                      <div 
                        key={skill.id}
                        class="group relative p-3 rounded-md border border-border/50 bg-muted/10 hover:bg-muted/30 hover:border-primary/30 transition-all duration-200"
                      >
                        <div class="flex items-start justify-between gap-2 mb-1">
                          <div class="font-medium text-sm text-foreground group-hover:text-primary transition-colors">
                            {skill.name}
                          </div>
                          <span class="text-[10px] px-1.5 py-0.5 rounded-full bg-muted/50 text-muted-foreground font-mono opacity-0 group-hover:opacity-100 transition-opacity">
                            L{Math.ceil(Math.random() * 8)} {/* Placeholder for Level until parsed */}
                          </span>
                        </div>
                        <div class="text-xs text-muted-foreground leading-relaxed line-clamp-2 group-hover:line-clamp-none transition-all">
                          {skill.description}
                        </div>
                        
                        {/* Action Bar (Slide up on hover) */}
                        <div class="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
                          <button 
                            class="p-1 hover:bg-primary/20 rounded text-[10px] text-primary"
                            title="Copy ID"
                            onClick$={() => navigator.clipboard.writeText(skill.id)}
                          >
                            📋
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
});
