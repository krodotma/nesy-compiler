import { component$, useVisibleTask$ } from '@builder.io/qwik';

const ART_THEMES = [
    "0fb64828691c74eb.css",
    "10099f28eda33175.css",
    "1e4e8c2645d1c2b3.css",
    "225cd3c56945aee5.css",
    "307d88c51eb5d2b5.css",
    "33448a8c072f922b.css",
    "561ec2e52a8b28a5.css",
    "66d3a781989942cf.css",
    "6f6b21f51a56d1e4.css",
    "79f8766297add503.css",
    "a4cc39aeb0f09297.css",
    "base_theme.css",
    "c5a5df7b26ab0f6e.css",
    "ce01a9642b1fdb12.css",
    "cebe0f7b44341c8e.css",
    "cyber_brutalist.css",
    "f01f23c2f571a567.css",
    "retro_terminal.css",
    "theme-minimal-1765988920.css",
    "theme-organic-1765988920.css"
];

export const ArtDeptInjector = component$(() => {
    useVisibleTask$(() => {
        // Only inject if no art theme is currently loaded
        if (document.getElementById('pluribus-art-theme')) return;

        // Random selection
        const randomTheme = ART_THEMES[Math.floor(Math.random() * ART_THEMES.length)];
        const themeUrl = `/art_dept/${randomTheme}`;

        const link = document.createElement('link');
        link.id = 'pluribus-art-theme';
        link.rel = 'stylesheet';
        link.href = themeUrl;

        // Append to head
        document.head.appendChild(link);

        console.log(`%c[ART DEPT] Injected: ${randomTheme}`, 'color: #ff00ff; font-weight: bold');
    });

    return null; // Headless component
});
