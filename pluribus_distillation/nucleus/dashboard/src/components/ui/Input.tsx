import { component$, type PropFunction } from '@builder.io/qwik';

// M3 Components - Input/TextField variants
import '@material/web/textfield/filled-text-field.js';
import '@material/web/textfield/outlined-text-field.js';
import '@material/web/icon/icon.js';

interface InputProps {
  label?: string;
  value?: string;
  placeholder?: string;
  type?: 'text' | 'password' | 'email' | 'number' | 'search' | 'textarea';
  error?: string;
  class?: string;
  icon?: string; // Leading icon
  disabled?: boolean;
  // Events
  onInput$?: PropFunction<(event: Event, el: HTMLInputElement) => void>;
  onChange$?: PropFunction<(event: Event, el: HTMLInputElement) => void>;
  onKeyDown$?: PropFunction<(event: KeyboardEvent, el: HTMLInputElement) => void>;
}

export const Input = component$((props: InputProps) => {
  const isTextArea = props.type === 'textarea';

  // Cast props to any to bypass strict type checking for web component events
  // @ts-ignore
  const inputProps: any = {
    label: props.label,
    value: props.value,
    placeholder: props.placeholder,
    'error-text': props.error,
    class: props.class,
    disabled: props.disabled,
    onInput$: props.onInput$,
    onChange$: props.onChange$,
    onKeyDown$: props.onKeyDown$,
  };

  if (isTextArea) {
    return (
      <md-outlined-text-field
        type="textarea"
        rows={3}
        {...inputProps}
      >
        {props.icon && <md-icon slot="leading-icon">{props.icon}</md-icon>}
      </md-outlined-text-field>
    );
  }

  // Search optimized
  if (props.type === 'search') {
    return (
      <md-outlined-text-field
        type="search"
        {...inputProps}
      >
        <md-icon slot="leading-icon">search</md-icon>
      </md-outlined-text-field>
    );
  }

  // Default Text Field (Filled)
  return (
    <md-filled-text-field
      type={props.type || 'text'}
      {...inputProps}
    >
      {props.icon && <md-icon slot="leading-icon">{props.icon}</md-icon>}
    </md-filled-text-field>
  );
});
