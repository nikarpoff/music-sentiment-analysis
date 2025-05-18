import classes from './ui.module.css';

export const DarkButton = ({ children, onClick, disabled = false }) => (
    <button onClick={onClick} className={classes.darkButton} disabled={disabled}>
        {children}
    </button>
);
