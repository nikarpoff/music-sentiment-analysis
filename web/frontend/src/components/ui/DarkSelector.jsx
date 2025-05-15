import styles from './ui.module.css';

export const DarkSelector = ({
    options = [],
    value,
    onChange,
    disabled = false,
    placeholder = 'Select…'
  }) => (
    <select
        className={styles.darkSelector}
        value={value}
        onChange={onChange}
        disabled={disabled}
        placeholder={placeholder}
    >
        {options.map(opt => (
            <option key={opt.value} value={opt.value}>
                {opt.label}
            </option>
        ))}
    </select>
);
  