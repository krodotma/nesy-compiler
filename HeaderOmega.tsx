/**
 * HeaderOmega: The Top Bread
 * ==========================
 * Persistent application header with:
 * - Glass morphism surfaces with theme-aware gradients
 * - 3D CSS rotating hamburger menu toggle
 * - Proper light/dark mode contrast
 */

import { component$, useVisibleTask$ } from '@builder.io/qwik';
import { VisionEye } from './components/VisionEye';

// ... (existing imports)

// ... inside HeaderOmega component ...

          <ThemeModeToggle />
          <GestaltPill mood={props.mood} entropy={props.entropy} />
          <VisionEye />
          <VNCAuthFab
            providerStatus={props.providerStatus}
            onOpen$={props.onOpenAuth$}
            inline={true}
          />
