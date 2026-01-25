import { component$, useSignal, useVisibleTask$, $ } from '@builder.io/qwik';

export interface VirtualListProps {
  items: any[];
  height: number;
  itemHeight: number;
  renderItem$: (item: any, index: number) => any;
  className?: string;
}

export const VirtualList = component$<VirtualListProps>(({ items, height, itemHeight, renderItem$, className }) => {
  const containerRef = useSignal<HTMLDivElement>();
  const scrollTop = useSignal(0);

  useVisibleTask$(({ track }) => {
    const el = track(() => containerRef.value);
    if (!el) return;

    const handleScroll = () => {
      scrollTop.value = el.scrollTop;
    };

    el.addEventListener('scroll', handleScroll, { passive: true });
    return () => el.removeEventListener('scroll', handleScroll);
  });

  const totalHeight = items.length * itemHeight;
  const startIndex = Math.floor(scrollTop.value / itemHeight);
  const visibleCount = Math.ceil(height / itemHeight);
  const endIndex = Math.min(items.length, startIndex + visibleCount + 5); // buffer
  const visibleItems = items.slice(startIndex, endIndex);
  const offsetY = startIndex * itemHeight;

  return (
    <div
      ref={containerRef}
      class={`overflow-auto relative ${className}`}
      style={{ height: `${height}px` }}
    >
      <div style={{ height: `${totalHeight}px`, position: 'relative' }}>
        <div style={{ transform: `translateY(${offsetY}px)`, position: 'absolute', top: 0, left: 0, width: '100%' }}>
          {visibleItems.map((item, i) => (
            <div key={startIndex + i} style={{ height: `${itemHeight}px` }}>
              {renderItem$(item, startIndex + i)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
});
