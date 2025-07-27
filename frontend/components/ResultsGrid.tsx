"use client"

import { useState, useEffect } from "react"
import { Star, Heart } from "lucide-react"

interface Product {
  id: number
  title: string
  description: string
  brand: string
  category: string
  price: number
  discount: number
  rating: number
  review_count: number
  stock: number
  tags: string[]
  sku: string
  thumbnail: string
  images: string[]
  shippingInformation: string
  returnPolicy: string
  minimumOrderQuantity: number
  dimensions: object
  warrantyInformation: string
  warehouse_city: string
  location: { lat: number; lon: number }
  meta: object
  created_at: string
  updated_at: string
}

interface ResultsGridProps {
  query: string
  filters: any
  sort: string
  userLat: number | null
  userLon: number | null
}

export default function ResultsGrid({ query, filters, sort, userLat, userLon }: ResultsGridProps) {
  const [products, setProducts] = useState<Product[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!query.trim()) {
      setProducts([])
      return
    }

    setLoading(true)

    // Real backend call
    const searchParams = new URLSearchParams({
      q: query,
      min_price: filters.priceMin.toString(),
      max_price: filters.priceMax.toString(),
      min_rating: filters.rating.toString(),
      brand: filters.brands.join(','),
      category: filters.categories.join(',')
    })

    fetch(`http://localhost:8000/search?${searchParams}`)
      .then(res => res.json())
      .then(data => {
        // Transform backend data to match frontend Product interface
        const transformedProducts: Product[] = data.map((item: any, index: number) => ({
          id: item.id || index + 1,
          title: item.title,
          description: item.title, // Backend doesn't have description, using title
          brand: item.title.split(' ')[0], // Extract brand from title
          category: "Electronics", // Default category
          price: item.price,
          discount: Math.floor(Math.random() * 20) + 5, // Random discount for demo
          rating: item.rating,
          review_count: Math.floor(Math.random() * 2000) + 100, // Random review count
          stock: Math.floor(Math.random() * 100) + 10, // Random stock
          tags: [query, "bestseller"],
          sku: `SKU${item.id || index + 1}`,
          thumbnail: item.image || `/placeholder.svg?height=200&width=200&query=${encodeURIComponent(item.title)}`,
          images: [item.image || `/placeholder.svg?height=400&width=400&query=${encodeURIComponent(item.title)}`],
          shippingInformation: "Free delivery",
          returnPolicy: "7 days return",
          minimumOrderQuantity: 1,
          dimensions: { width: 10, height: 15, depth: 2 },
          warrantyInformation: "1 year warranty",
          warehouse_city: ["Mumbai", "Delhi", "Bangalore", "Chennai"][Math.floor(Math.random() * 4)],
          location: { lat: 28.6139, lon: 77.209 }, // Default to Delhi
          meta: {},
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        }))
        setProducts(transformedProducts)
      })
      .catch(error => {
        console.error('Error fetching search results:', error)
        // Fallback to dummy data if backend is not available
    setTimeout(() => {
      const dummyProducts: Product[] = [
        {
          id: 1,
          title: `${query} Premium Edition`,
          description: `High quality ${query} with premium features and excellent build quality`,
          brand: "Apple",
          category: "Electronics",
          price: 25999,
          discount: 15,
          rating: 4.5,
          review_count: 1250,
          stock: 50,
          tags: [query, "premium", "bestseller"],
          sku: "SKU001",
          thumbnail: "/placeholder.svg?height=200&width=200&query=" + encodeURIComponent(query + " premium"),
          images: ["/placeholder.svg?height=400&width=400&query=" + encodeURIComponent(query)],
          shippingInformation: "Free delivery",
          returnPolicy: "7 days return",
          minimumOrderQuantity: 1,
          dimensions: { width: 10, height: 15, depth: 2 },
          warrantyInformation: "1 year warranty",
          warehouse_city: "Mumbai",
          location: { lat: 19.076, lon: 72.8777 },
          meta: {},
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
        {
          id: 2,
          title: `${query} Standard`,
          description: `Reliable ${query} with good features at affordable price`,
          brand: "Samsung",
          category: "Electronics",
          price: 15999,
          discount: 10,
          rating: 4.2,
          review_count: 890,
          stock: 75,
          tags: [query, "standard", "value"],
          sku: "SKU002",
          thumbnail: "/placeholder.svg?height=200&width=200&query=" + encodeURIComponent(query + " standard"),
          images: ["/placeholder.svg?height=400&width=400&query=" + encodeURIComponent(query)],
          shippingInformation: "Free delivery",
          returnPolicy: "7 days return",
          minimumOrderQuantity: 1,
          dimensions: { width: 8, height: 12, depth: 1.5 },
          warrantyInformation: "1 year warranty",
          warehouse_city: "Delhi",
          location: { lat: 28.6139, lon: 77.209 },
          meta: {},
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
        {
          id: 3,
          title: `${query} Pro Max`,
          description: `Professional grade ${query} with advanced features`,
          brand: "Sony",
          category: "Electronics",
          price: 35999,
          discount: 20,
          rating: 4.7,
          review_count: 2100,
          stock: 25,
          tags: [query, "pro", "professional"],
          sku: "SKU003",
          thumbnail: "/placeholder.svg?height=200&width=200&query=" + encodeURIComponent(query + " pro"),
          images: ["/placeholder.svg?height=400&width=400&query=" + encodeURIComponent(query)],
          shippingInformation: "Free delivery",
          returnPolicy: "10 days return",
          minimumOrderQuantity: 1,
          dimensions: { width: 12, height: 18, depth: 3 },
          warrantyInformation: "2 year warranty",
          warehouse_city: "Bangalore",
          location: { lat: 12.9716, lon: 77.5946 },
          meta: {},
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
        {
          id: 4,
          title: `${query} Lite`,
          description: `Budget-friendly ${query} with essential features`,
          brand: "Realme",
          category: "Electronics",
          price: 8999,
          discount: 5,
          rating: 3.9,
          review_count: 450,
          stock: 100,
          tags: [query, "lite", "budget"],
          sku: "SKU004",
          thumbnail: "/placeholder.svg?height=200&width=200&query=" + encodeURIComponent(query + " lite"),
          images: ["/placeholder.svg?height=400&width=400&query=" + encodeURIComponent(query)],
          shippingInformation: "Free delivery",
          returnPolicy: "7 days return",
          minimumOrderQuantity: 1,
          dimensions: { width: 6, height: 10, depth: 1 },
          warrantyInformation: "1 year warranty",
          warehouse_city: "Chennai",
          location: { lat: 13.0827, lon: 80.2707 },
          meta: {},
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
      ]
      setProducts(dummyProducts)
    }, 500)
      })
      .finally(() => setLoading(false))
  }, [query, filters, sort, userLat, userLon])

  const addToCart = (product: Product) => {
    const cart = JSON.parse(localStorage.getItem("cart") || "[]")
    const existingItemIndex = cart.findIndex((item: any) => item.id === product.id)

    if (existingItemIndex > -1) {
      // Item exists, increase quantity
      cart[existingItemIndex].quantity += 1
    } else {
      // New item, add to cart
      const discountedPrice = product.price - (product.price * product.discount) / 100
      cart.push({
        id: product.id,
        title: product.title,
        price: Math.round(discountedPrice),
        quantity: 1,
        thumbnail: product.thumbnail,
      })
    }

    localStorage.setItem("cart", JSON.stringify(cart))

    // Trigger storage event for cart count update
    window.dispatchEvent(
      new StorageEvent("storage", {
        key: "cart",
        newValue: JSON.stringify(cart),
      }),
    )

    alert(`${product.title} added to cart!`)
  }

  const calculateDeliveryDays = (warehouseCity: string) => {
    // Simple logic based on warehouse city
    const deliveryMap: { [key: string]: number } = {
      Mumbai: 2,
      Delhi: 1,
      Bangalore: 3,
      Chennai: 4,
    }
    return deliveryMap[warehouseCity] || 3
  }

  if (loading) {
    return (
      <div className="p-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="bg-white p-4 rounded shadow animate-pulse">
              <div className="bg-gray-200 h-48 rounded mb-4"></div>
              <div className="bg-gray-200 h-4 rounded mb-2"></div>
              <div className="bg-gray-200 h-4 rounded w-3/4"></div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (!query) {
    return (
      <div className="p-8 text-center text-gray-500">
        <p>Search for products to see results</p>
      </div>
    )
  }

  return (
    <div className="p-2 sm:p-4">
      <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 sm:gap-4">
        {products.map((product) => {
          const discountedPrice = product.price - (product.price * product.discount) / 100
          const deliveryDays = calculateDeliveryDays(product.warehouse_city)

          return (
            <div key={product.id} className="bg-white rounded shadow hover:shadow-lg transition-shadow p-2 sm:p-4">
              <div className="relative mb-2 sm:mb-4">
                <img
                  src={product.thumbnail || "/placeholder.svg"}
                  alt={product.title}
                  loading="lazy"
                  className="w-full h-32 sm:h-48 object-cover rounded"
                />
                <button className="absolute top-1 right-1 sm:top-2 sm:right-2 p-1 bg-white rounded-full shadow">
                  <Heart size={12} className="sm:w-4 sm:h-4 text-gray-400" />
                </button>
                {product.discount > 0 && (
                  <span className="absolute top-1 left-1 sm:top-2 sm:left-2 bg-green-500 text-white px-1 sm:px-2 py-0.5 sm:py-1 text-xs rounded">
                    {product.discount}% off
                  </span>
                )}
              </div>

              <h3 className="font-medium text-xs sm:text-sm mb-1 sm:mb-2 line-clamp-2">{product.title}</h3>

              <div className="flex items-center mb-1 sm:mb-2">
                <div className="flex items-center bg-green-500 text-white px-1 py-0.5 rounded text-xs">
                  <span className="text-xs">{product.rating}</span>
                  <Star size={8} className="sm:w-2.5 sm:h-2.5 ml-1 fill-current" />
                </div>
                <span className="text-xs text-gray-500 ml-1 sm:ml-2">({product.review_count})</span>
              </div>

              <div className="mb-1 sm:mb-2">
                <span className="text-sm sm:text-lg font-semibold">
                  ₹{Math.round(discountedPrice).toLocaleString()}
                </span>
                {product.discount > 0 && (
                  <span className="text-xs sm:text-sm text-gray-500 line-through ml-1 sm:ml-2">
                    ₹{product.price.toLocaleString()}
                  </span>
                )}
              </div>

              <div className="text-xs text-gray-600 mb-2 sm:mb-3">{deliveryDays} days</div>

              <div className="space-y-1 sm:space-y-2">
                <button
                  onClick={() => addToCart(product)}
                  className="w-full bg-[#ff9f00] hover:bg-[#e68900] text-white py-1 sm:py-2 px-2 sm:px-4 rounded text-xs sm:text-sm font-medium transition-colors"
                >
                  Add to Cart
                </button>
                <button className="w-full bg-[#fb641b] hover:bg-[#e55a1b] text-white py-1 sm:py-2 px-2 sm:px-4 rounded text-xs sm:text-sm font-medium transition-colors">
                  Buy Now
                </button>
              </div>
            </div>
          )
        })}
      </div>

      {products.length === 0 && query && (
        <div className="text-center py-8 text-gray-500">
          <p className="text-sm sm:text-base">No products found for "{query}"</p>
          <p className="text-xs sm:text-sm mt-2">Try searching with different keywords</p>
        </div>
      )}
    </div>
  )
}
