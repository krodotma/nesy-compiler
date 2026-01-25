declare module 'packery';
declare module 'draggabilly';
declare module 'p5' {
  class p5 {
    constructor(sketch: (p: p5) => void, node?: HTMLElement);
    [key: string]: any;
  }
  type Color = any;
  export default p5;
  export { p5, Color };
}
