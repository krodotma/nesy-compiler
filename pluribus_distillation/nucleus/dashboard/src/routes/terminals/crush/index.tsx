import { component$ } from '@builder.io/qwik';
import { CrushTerminal } from '../../../components/terminal/CrushTerminal';

export default component$(() => {
  return (
    <div class="h-screen w-screen flex flex-col min-h-0 p-2 bg-[#0c0c0e]">
      <CrushTerminal />
    </div>
  );
});
