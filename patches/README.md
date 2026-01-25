# Deployment Instructions

## Files Prepared

1. `LoadingOverlay.tsx` → `/pluribus/nucleus/dashboard/src/components/LoadingOverlay.tsx`
2. `loading-overlay.css` → Append to `/pluribus/nucleus/dashboard/src/global.css`

## Apply Patches

Once VPS SSH is back:

```bash
# 1. Copy LoadingOverlay component
scp /Users/kroma/pluribus/patches/LoadingOverlay.tsx root@kroma.live:/pluribus/nucleus/dashboard/src/components/

# 2. Append CSS to global.css
cat /Users/kroma/pluribus/patches/loading-overlay.css | ssh root@kroma.live 'cat >> /pluribus/nucleus/dashboard/src/global.css'

# 3. Modify root.tsx to add LoadingOverlay import and usage
ssh root@kroma.live 'cat /pluribus/nucleus/dashboard/src/root.tsx'
# Then apply the root.tsx patch below

# 4. Set __pluribusReady in markHydrationEnd
# In load-timing.ts, add: window.__pluribusReady = true;
```

## root.tsx Modifications

Add import at top:
```tsx
import { LoadingOverlay } from './components/LoadingOverlay';
```

Add component as first child of `<body>`:
```tsx
<body lang="en" class="bg-background text-foreground">
  <LoadingOverlay />
  ...
```

## load-timing.ts Modification

In `markHydrationEnd()` function, add:
```typescript
export function markHydrationEnd(): void {
  hydrationEndMark = performance.now();
  window.__pluribusReady = true;  // ADD THIS LINE
}
```
