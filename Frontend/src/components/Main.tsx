// import React, { useState } from 'react';
// import "./Card.css"

// const Cards: React.FC = () => {
//     const [image, setImage] = useState<string | ArrayBuffer | null>(null);

//     const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
//         const file = event.target.files?.[0];
//         if (file) {
//             const reader = new FileReader();
//             reader.onloadend = () => {
//                 setImage(reader.result);
//             };
//             reader.readAsDataURL(file);
//         }
//     };

//     const handleFeatureClick = (feature: string) => {
//         alert(`Feature ${feature} applied!`);
//         // Implement the feature's functionality here
//     };

//     return (
//         <div className="cards-container">
//             <div className="upload-section">
//                 <input 
//                     type="file" 
//                     accept="image/*" 
//                     onChange={handleImageUpload} 
//                     className="upload-input"
//                 />
//             </div>
//             {image && (
//                 <div className="image-section">
//                     <img src={image as string} alt="Uploaded" className="uploaded-image" />
//                     <div className="buttons-container">
//                         <button 
//                             className="feature-button" 
//                             onClick={() => handleFeatureClick('Symmetry')}>
//                             Symmetry
//                         </button>
//                         <button 
//                             className="feature-button" 
//                             onClick={() => handleFeatureClick('Regularized')}>
//                             Regularized
//                         </button>
//                         <button 
//                             className="feature-button" 
//                             onClick={() => handleFeatureClick('Plannerized')}>
//                             Plannerized
//                         </button>
//                     </div>
//                 </div>
//             )}
//         </div>
//     );
// };

// export default Cards;

import React, { useState } from 'react';
import "./Card.css";

const Cards: React.FC = () => {
    const [image, setImage] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | ArrayBuffer | null>(null);

    // Function to handle image upload
    const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setImage(file);
            const reader = new FileReader();
            reader.onloadend = () => {
                setImagePreview(reader.result);
            };
            reader.readAsDataURL(file);
        }
    };

    // Function to handle feature button clicks
    const handleFeatureClick = (feature: string) => {
        if (!image) {
            console.error('No image file provided');
            return;
        }

        const formData = new FormData();
        formData.append('file', image);
        formData.append('feature', feature);

        // Send the image and selected feature to the Flask backend
        fetch('http://localhost:5000/process_image', {
            method: 'POST',
            body: formData,
        })
        .then((response) => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.blob();
        })
        .then((blob) => {
            const url = URL.createObjectURL(blob);
            // Open the processed image in a new window
            window.open(url);
        })
        .catch((error) => console.error('Error:', error));
    };

    return (
        <div className="cards-container">
            <div className="upload-section">
                <input 
                    type="file" 
                    accept="image/*" 
                    onChange={handleImageUpload} 
                    className="upload-input"
                />
            </div>
            {imagePreview && (
                <div className="image-section">
                    <img src={imagePreview as string} alt="Uploaded" className="uploaded-image" />
                    <div className="buttons-container">
                        <button 
                            className="feature-button" 
                            onClick={() => handleFeatureClick('Symmetry')}>
                            Symmetry
                        </button>
                        <button 
                            className="feature-button" 
                            onClick={() => handleFeatureClick('Regularized')}>
                            Plannerized
                        </button>
                        <button 
                            className="feature-button" 
                            onClick={() => handleFeatureClick('Plannerized')}>
                            Regularized
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Cards;
