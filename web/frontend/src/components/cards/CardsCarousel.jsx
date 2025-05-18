import React, { useState } from 'react';
import classes from './cards.module.css';

export const CardCarousel = ({ children, onCardChange }) => {
    const total = React.Children.count(children);
    const [currentIndex, setCurrentIndex] = useState(0);

    const prevCard = () => {
        const newIndex = (currentIndex - 1 + total) % total;
        onCardChange(children[newIndex].key);
        setCurrentIndex(newIndex);
    }

    const nextCard = () => {
        const newIndex = (currentIndex + 1) % total
        onCardChange(children[newIndex].key);
        setCurrentIndex(newIndex);
    }

    return (
        <div className={classes.carouselContainer}>
            <button
                className={`${classes.navButton} ${classes.prev}`}
                onClick={prevCard}
            >
                &lt;
            </button>

            <div
                className={classes.cardWrapper}
                style={{ transform: `translateX(-${currentIndex * 100}%)` }}
            >
                {React.Children.map(children, (child, index) => (
                    <div key={index} className={classes.card}>
                        {child}
                    </div>
                ))}
            </div>

            <button
                className={`${classes.navButton} ${classes.next}`}
                onClick={nextCard}
            >
                &gt;
            </button>
        </div>
    );
};
