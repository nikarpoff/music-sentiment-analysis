import { useEffect } from 'react';
import classes from './ui.module.css';
import { DarkButton } from './DarkButton';
import { EmotionRadarChart } from './DarkRadar';

// Базовые цвета для эмоций в формате [R, G, B]
const EMOTION_COLORS = {
    sad:      [0, 123, 255],    // насыщенный синий
    relaxing: [102, 204, 255],  // небесно-голубой
    happy:    [0, 200, 0],      // ярко-зеленый
    energetic:[255, 165, 0],    // оранжевый
};

/**
 * Смешивает цвета на основе вероятностей эмоций
 * @param {{[key: string]: number}} data — вероятности эмоций
 * @param {{[key: string]: number[]}} colors — базовые цвета эмоций
 * @returns {string} — итоговый CSS rgb цвет
 */
function mixBackgroundColor(data, colors) {
    let r = 0, g = 0, b = 0;
    Object.entries(data).forEach(([emotion, weight]) => {
        const [cr, cg, cb] = colors[emotion] || [0, 0, 0];
        r += cr * weight;
        g += cg * weight;
        b += cb * weight;
    });
    const toByte = x => Math.min(255, Math.max(0, Math.round(x)));
    return `rgb(${toByte(r)}, ${toByte(g)}, ${toByte(b)})`;
}

export const AlertModal = ({ isOpen, onClose, text }) => {
    useEffect(() => {
        document.body.style.overflow = isOpen ? 'hidden' : 'auto';
    }, [isOpen]);

    if (!isOpen) return null;

    return (
        <div
            className={`${classes.overlay} ${isOpen ? classes.overlayOpen : ''}`}
            onClick={e => e.target === e.currentTarget && onClose()}
        >
            <div className={`${classes.modalContent} ${isOpen ? classes.modalContentOpen : ''}`}>  {/* фон не меняется здесь */}
                <div className={classes.modalBody}>
                    {text}
                </div>

                <DarkButton onClick={onClose}>
                    Закрыть
                </DarkButton>
            </div>
        </div>
    );
};

export const ResultModal = ({ isOpen, onClose, predict, probs }) => {
    useEffect(() => {
        document.body.style.overflow = isOpen ? 'hidden' : 'auto';
    }, [isOpen]);

    if (!isOpen) return null;

    const translation = {
        sad:       'грустная',
        happy:     'веселая',
        energetic: 'энергичная',
        relaxing:  'расслабляющая',
    };

    // Вычисляем цвет фона на основе вероятностей
    const moodColor = mixBackgroundColor(probs, EMOTION_COLORS);

    return (
        <div
            className={`${classes.overlay} ${isOpen ? classes.overlayOpen : ''}`}
            onClick={e => e.target === e.currentTarget && onClose()}
        >
            <div
                className={`${classes.modalContent} ${isOpen ? classes.modalContentOpen : ''}`}
                style={{ borderColor: moodColor }}
            >
                <div className={classes.modalBody}>
                    <p style={{ margin: 0, padding: 0 }}>
                        Ваша музыка&nbsp;
                        <span style={{ color: moodColor, padding: 0, margin: 0 }}>
                            {translation[predict]}
                        </span>
                    </p>
                    <EmotionRadarChart data={probs} />
                </div>

                <DarkButton onClick={onClose}>
                    Назад
                </DarkButton>
            </div>
        </div>
    );
};
