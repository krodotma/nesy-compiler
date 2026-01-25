/**
 * Tailwind CSS Configuration (Material Web 3 / Deep Neon Edition)
 *
 * Updated to support CSS Color Module Level 4 (OKLCH) and Material 3 System Tokens.
 * Removes legacy HSL wrappers to allow direct variable usage.
 */

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ['class'],
  content: [
    './src/**/*.{ts,tsx,js,jsx,html}',
    './src/components/**/*.{ts,tsx}',
    './src/routes/**/*.{ts,tsx}',
  ],
  theme: {
    container: {
      center: true,
      padding: '2rem',
      screens: {
        '2xl': '1400px',
      },
    },
    extend: {
      colors: {
        // Core M3 System Tokens (Mapped to CSS variables in material-tokens.css)
        md: {
          primary: 'var(--md-sys-color-primary)',
          'on-primary': 'var(--md-sys-color-on-primary)',
          'primary-container': 'var(--md-sys-color-primary-container)',
          'on-primary-container': 'var(--md-sys-color-on-primary-container)',

          secondary: 'var(--md-sys-color-secondary)',
          'on-secondary': 'var(--md-sys-color-on-secondary)',
          'secondary-container': 'var(--md-sys-color-secondary-container)',
          'on-secondary-container': 'var(--md-sys-color-on-secondary-container)',

          tertiary: 'var(--md-sys-color-tertiary)',
          'on-tertiary': 'var(--md-sys-color-on-tertiary)',
          'tertiary-container': 'var(--md-sys-color-tertiary-container)',
          'on-tertiary-container': 'var(--md-sys-color-on-tertiary-container)',

          error: 'var(--md-sys-color-error)',
          'on-error': 'var(--md-sys-color-on-error)',
          'error-container': 'var(--md-sys-color-error-container)',
          'on-error-container': 'var(--md-sys-color-on-error-container)',

          surface: 'var(--md-sys-color-surface)',
          'on-surface': 'var(--md-sys-color-on-surface)',
          'surface-variant': 'var(--md-sys-color-surface-variant)',
          'on-surface-variant': 'var(--md-sys-color-on-surface-variant)',

          outline: 'var(--md-sys-color-outline)',
          'outline-variant': 'var(--md-sys-color-outline-variant)',

          'inverse-surface': 'var(--md-sys-color-inverse-surface)',
          'inverse-on-surface': 'var(--md-sys-color-inverse-on-surface)',
          'inverse-primary': 'var(--md-sys-color-inverse-primary)',

          scrim: 'var(--md-sys-color-scrim)',
          shadow: 'var(--md-sys-color-shadow)',

          // Surface Containers (Elevation)
          'surface-container-lowest': 'var(--md-sys-color-surface-container-lowest)',
          'surface-container-low': 'var(--md-sys-color-surface-container-low)',
          'surface-container': 'var(--md-sys-color-surface-container)',
          'surface-container-high': 'var(--md-sys-color-surface-container-high)',
          'surface-container-highest': 'var(--md-sys-color-surface-container-highest)',
        },

        // Legacy / Shadcn Compat (Remapped to avoid HSL breakage)
        border: 'var(--border)',
        input: 'var(--input)',
        ring: 'var(--ring)',
        background: 'var(--background)',
        foreground: 'var(--foreground)',
        primary: {
          DEFAULT: 'var(--primary)',
          foreground: 'var(--primary-foreground)',
        },
        secondary: {
          DEFAULT: 'var(--secondary)',
          foreground: 'var(--secondary-foreground)',
        },
        destructive: {
          DEFAULT: 'var(--destructive)',
          foreground: 'var(--destructive-foreground)',
        },
        muted: {
          DEFAULT: 'var(--muted)',
          foreground: 'var(--muted-foreground)',
        },
        accent: {
          DEFAULT: 'var(--accent)',
          foreground: 'var(--accent-foreground)',
        },
        popover: {
          DEFAULT: 'var(--popover)',
          foreground: 'var(--popover-foreground)',
        },
        card: {
          DEFAULT: 'var(--card)',
          foreground: 'var(--card-foreground)',
        },

        // ART DEPARTMENT (V2 System) - Deep Neon
        art: {
          start: 'var(--art-bg-start)',
          end: 'var(--art-bg-end)',
          surface: 'var(--art-surface)',
        },
        // Status colors (unified with TUI)
        status: {
          success: 'var(--status-success)',
          warning: 'var(--status-warning)',
          error: 'var(--status-error)',
          info: 'var(--status-info)',
          idle: 'var(--status-idle)',
        },
        // Flow mode colors
        mode: {
          monitor: 'hsl(var(--mode-monitor))', // Keeping HSL if these are explicitly legacy
          auto: 'hsl(var(--mode-auto))',
        },
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      keyframes: {
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height)' },
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height)' },
          to: { height: '0' },
        },
        pulse: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.5' },
        },
        'neon-pulse': {
          '0%, 100%': { filter: 'drop-shadow(0 0 5px var(--md-sys-color-primary))' },
          '50%': { filter: 'drop-shadow(0 0 15px var(--md-sys-color-primary))' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
        pulse: 'pulse 1.5s ease-in-out infinite',
        'neon-pulse': 'neon-pulse 2s ease-in-out infinite',
        shimmer: 'shimmer 2s linear infinite',
      },
      backdropBlur: {
        xs: '2px',
        glass: '16px',
      },
      boxShadow: {
        'neon': '0 0 5px var(--md-sys-color-primary), 0 0 20px var(--md-sys-color-primary-container)',
        'glass': '0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05)',
        'glass-glow': '0 0 20px -5px var(--glass-accent-cyan-glow)',
        'glass-glow-lg': '0 0 30px -3px var(--glass-accent-cyan-glow)',
      },

      // Phase 3: Dynamic Gradient Utilities
      backgroundImage: {
        // Aurora gradients
        'glass-aurora': 'var(--glass-gradient-aurora)',
        'glass-conic-neon': 'var(--glass-gradient-conic-neon)',
        'glass-conic-soft': 'var(--glass-gradient-conic-soft)',

        // Status gradients
        'glass-status-healthy': 'linear-gradient(135deg, oklch(70% 0.2 145 / 0.2), oklch(75% 0.15 180 / 0.1))',
        'glass-status-warning': 'linear-gradient(135deg, oklch(80% 0.18 85 / 0.2), oklch(75% 0.15 60 / 0.1))',
        'glass-status-error': 'linear-gradient(135deg, oklch(60% 0.22 25 / 0.2), oklch(55% 0.2 0 / 0.1))',
        'glass-status-info': 'linear-gradient(135deg, oklch(70% 0.18 230 / 0.2), oklch(65% 0.15 200 / 0.1))',

        // Progress gradients
        'glass-progress': 'linear-gradient(90deg, var(--glass-accent-cyan), var(--glass-accent-purple), var(--glass-accent-magenta))',
        'glass-wip-low': 'linear-gradient(90deg, oklch(60% 0.22 25), oklch(65% 0.2 45))',
        'glass-wip-medium': 'linear-gradient(90deg, oklch(75% 0.18 85), oklch(70% 0.2 200))',
        'glass-wip-high': 'linear-gradient(90deg, oklch(70% 0.2 200), oklch(70% 0.2 280))',
        'glass-wip-complete': 'linear-gradient(90deg, oklch(70% 0.2 145), oklch(75% 0.18 165))',

        // Depth gradients
        'glass-depth': 'linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02))',
        'glass-depth-reverse': 'linear-gradient(0deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02))',

        // Mesh gradient
        'glass-mesh': `
          radial-gradient(at 40% 20%, var(--glass-accent-cyan-subtle) 0px, transparent 50%),
          radial-gradient(at 80% 0%, var(--glass-accent-magenta-subtle) 0px, transparent 50%),
          radial-gradient(at 0% 50%, var(--glass-accent-purple-bg) 0px, transparent 50%)
        `,
      },

      // Phase 3: Grey scale colors
      colors: {
        ...{
          'glass-grey': {
            50: 'oklch(95% 0.01 240)',
            100: 'oklch(90% 0.01 240)',
            200: 'oklch(80% 0.01 240)',
            300: 'oklch(70% 0.01 240)',
            400: 'oklch(60% 0.01 240)',
            500: 'oklch(50% 0.01 240)',
            600: 'oklch(40% 0.01 240)',
            700: 'oklch(30% 0.01 240)',
            800: 'oklch(20% 0.01 240)',
            900: 'oklch(15% 0.01 240)',
          },
          'glass-highlight': {
            primary: 'var(--glass-highlight-primary)',
            success: 'var(--glass-highlight-success)',
            warning: 'var(--glass-highlight-warning)',
            error: 'var(--glass-highlight-error)',
            info: 'var(--glass-highlight-info)',
            special: 'var(--glass-highlight-special)',
          },
          'glass-data': {
            1: 'var(--glass-data-1)',
            2: 'var(--glass-data-2)',
            3: 'var(--glass-data-3)',
            4: 'var(--glass-data-4)',
            5: 'var(--glass-data-5)',
            6: 'var(--glass-data-6)',
            7: 'var(--glass-data-7)',
            8: 'var(--glass-data-8)',
          },
        },
      },
    },
  },
  plugins: [],
};
