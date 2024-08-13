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
    const [result, setResult] = useState<string | null>(null); // State to store the result
    const [processedImage, setProcessedImage] = useState<string | null>(null); // State to store processed image URL

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
        fetch('https://curvetopia-ljro.onrender.com/process_image', {
            method: 'POST',
            body: formData,
        })
        .then((response) => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return feature === 'Symmetry' ? response.json() : response.blob();
        })
        .then((data) => {
            if (feature === 'Symmetry') {
                setResult(JSON.stringify(data, null, 2)); // Convert the symmetry result to formatted JSON string
                setProcessedImage(null); // Clear processed image if it's not an image result
            } else {
                const url = URL.createObjectURL(data as Blob);
                setProcessedImage(url); // Set the processed image URL
                setResult(null); // Clear text result if it's not a text response
            }
        })
        .catch((error) => console.error('Error:', error));
    };

    return (
        <>
        <div className="flex-container">
            <h1>Curvetopia</h1>
        </div>
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
                            Planerized 
                        </button>
                        <button 
                            className="feature-button" 
                            onClick={() => handleFeatureClick('Plannerized')}>
                            Regularized
                        </button>
                    </div>
                </div>
            )}
            {result && (
                <div className="result-section">
                    <h3 style={{ color:'black', fontSize:'25px'}}>Result:</h3>
                    <pre style={{ backgroundColor: ' #32CD32', padding: '20px', borderRadius: '8px', color:'white', fontSize:'15px'}}>
                        <code>{result}</code>
                    </pre>
                </div>
            )}
            {processedImage && (
                <div className="processed-image-section">
                    <h3 style={{color:'black', fontSize:'25px'}}>Processed Image:</h3>
                    <img src={processedImage} alt="Processed" className="processed-image" />
                </div>
            )}
        </div>
        </>
    );
};

export default Cards;

